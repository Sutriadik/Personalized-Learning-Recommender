import os
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import TfidfTransformer

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")

MODEL_DIR = "models"


# ---- Data Loaders ----

def load_ratings():
    return pd.read_csv("data/ratings.csv")


def load_course_sims():
    return pd.read_csv("data/sim.csv")


def load_courses():
    df = pd.read_csv("data/course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("data/courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("data/ratings.csv", index=False)
        return new_id


# ---- Helper Functions ----

def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def build_course_profile_matrix():
    """Pivot BOW data into a courses x tokens feature matrix."""
    bow_df = load_bow()
    return bow_df.pivot_table(index='doc_id', columns='token', values='bow', fill_value=0)


def build_tfidf_course_matrix():
    """
    Build TF-IDF weighted course feature matrix.
    TF-IDF re-weights BOW counts so common tokens across all courses get
    lower weight, making course profiles more discriminative.
    """
    raw = build_course_profile_matrix()
    tfidf = TfidfTransformer(smooth_idf=True, sublinear_tf=True)
    tfidf_values = tfidf.fit_transform(raw.values).toarray()
    return pd.DataFrame(tfidf_values, index=raw.index, columns=raw.columns), tfidf


def build_user_course_matrix():
    """Pivot ratings into a users x courses interaction matrix (raw ratings)."""
    ratings_df = load_ratings()
    return ratings_df.pivot_table(index='user', columns='item', values='rating', fill_value=0)


def build_binary_user_course_matrix():
    """
    Binary interaction matrix: 1 = enrolled (any rating), 0 = not enrolled.
    Better for NMF/clustering with 95% sparse data — avoids dilution from
    the narrow 2.0-3.0 rating range.
    """
    ratings_df = load_ratings()
    return ratings_df.pivot_table(
        index='user', columns='item', values='rating',
        fill_value=0, aggfunc=lambda x: 1
    )


def build_weighted_user_course_matrix():
    """
    Weighted matrix: 3.0→1.0, 2.0→0.5, 0→0.
    Differentiates completed vs audited courses while still being non-negative.
    """
    ucm = build_user_course_matrix()
    weighted = ucm.copy()
    weighted[weighted == 3.0] = 1.0
    weighted[weighted == 2.0] = 0.5
    return weighted


def get_user_vector(enrolled_course_ids, course_columns, weight=1.0):
    """Build a sparse user vector aligned to training course columns."""
    vec = np.zeros(len(course_columns))
    col_index = {c: i for i, c in enumerate(course_columns)}
    for course in enrolled_course_ids:
        if course in col_index:
            vec[col_index[course]] = weight
    return vec


def cosine_sim(user_vec, matrix_values):
    """Cosine similarity between a single vector and each row of a matrix."""
    user_norm = np.linalg.norm(user_vec)
    if user_norm == 0:
        return np.zeros(matrix_values.shape[0])
    row_norms = np.linalg.norm(matrix_values, axis=1) + 1e-10
    return matrix_values.dot(user_vec) / (row_norms * user_norm)


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    """Original Course Similarity logic — kept unchanged."""
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    res = {}
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    return dict(sorted(res.items(), key=lambda item: item[1], reverse=True))


def _build_embedding_dataset(ucm, W, H):
    """
    Build (X, y) from NMF user/course factor matrices.
    X = concat(user_factors, course_factors)
    y = rating (normalised to [0,1])
    """
    X, y = [], []
    rating_max = ucm.values.max()
    for i in range(len(ucm.index)):
        for j in range(len(ucm.columns)):
            rating = ucm.iloc[i, j]
            if rating > 0:
                X.append(np.concatenate([W[i], H[:, j]]))
                y.append(rating / rating_max)   # normalise to [0,1]
    return np.array(X), np.array(y)


# ---- Model Training ----

def train(model_name, params):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if model_name == models[0]:
        # Course Similarity — no training needed
        pass

    elif model_name == models[1]:
        # User Profile — no training needed (TF-IDF built at predict time)
        pass

    elif model_name == models[2]:
        # Clustering
        # Uses BINARY matrix so KMeans clusters on enrollment patterns,
        # not diluted fractional ratings.
        cluster_no = params.get('cluster_no', 20)
        ucm = build_binary_user_course_matrix()
        kmeans = KMeans(n_clusters=cluster_no, random_state=42, n_init=10)
        kmeans.fit(ucm.values)
        joblib.dump((kmeans, ucm), os.path.join(MODEL_DIR, 'clustering.pkl'))

    elif model_name == models[3]:
        # Clustering with PCA
        # Scales → PCA (keep 95% variance) → KMeans on binary matrix.
        cluster_no = params.get('cluster_no', 20)
        ucm = build_binary_user_course_matrix()
        scaler = StandardScaler()
        X = scaler.fit_transform(ucm.values)
        # Keep enough components to explain 95% of variance, cap at 50
        pca_full = PCA(random_state=42)
        pca_full.fit(X)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = min(int(np.searchsorted(cumvar, 0.95)) + 1, 50, X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=cluster_no, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        joblib.dump((kmeans, pca, scaler, ucm), os.path.join(MODEL_DIR, 'clustering_pca.pkl'))

    elif model_name == models[4]:
        # KNN
        # Cosine distance on L2-normalised binary vectors: focuses on
        # enrollment pattern similarity, not vector magnitude.
        ucm = build_binary_user_course_matrix()
        X = normalize(ucm.values, norm='l2')
        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(X)
        joblib.dump((knn, ucm), os.path.join(MODEL_DIR, 'knn.pkl'))

    elif model_name == models[5]:
        # NMF
        # Trained on WEIGHTED matrix (1.0 = completed, 0.5 = audited).
        # sublinear_tf-style weight helps NMF handle the narrow rating range.
        n_components = params.get('n_components', 15)
        ucm = build_weighted_user_course_matrix()
        nmf = NMF(n_components=n_components, init='nndsvda',
                  random_state=42, max_iter=1000, l1_ratio=0.5)
        W = nmf.fit_transform(ucm.values)
        H = nmf.components_
        joblib.dump((nmf, W, H, ucm), os.path.join(MODEL_DIR, 'nmf.pkl'))

    elif model_name == models[6]:
        # Neural Network
        # Hybrid: user features = NMF factors + TF-IDF user profile vector (via SVD).
        # Course features = NMF factors + TF-IDF course vector (via SVD).
        # Richer input → more relevant recommendations.
        n_components = params.get('n_components', 15)
        ucm = build_weighted_user_course_matrix()
        # Collaborative part: NMF on interaction matrix
        nmf = NMF(n_components=n_components, init='nndsvda',
                  random_state=42, max_iter=1000, l1_ratio=0.5)
        W = nmf.fit_transform(ucm.values)
        H = nmf.components_
        # Content part: compress TF-IDF course matrix via TruncatedSVD
        tfidf_matrix, _ = build_tfidf_course_matrix()
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        course_svd = svd.fit_transform(tfidf_matrix.values)   # shape: (n_courses, n_components)
        course_svd_map = {cid: course_svd[i] for i, cid in enumerate(tfidf_matrix.index)}
        # Build training set
        X_train, y_train = _build_embedding_dataset(ucm, W, H)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=300,
                           random_state=42, early_stopping=True, validation_fraction=0.1)
        mlp.fit(X_train, y_train)
        joblib.dump((mlp, nmf, H, ucm, svd, course_svd_map, scaler),
                    os.path.join(MODEL_DIR, 'neural_network.pkl'))

    elif model_name == models[7]:
        # Regression with Embedding Features
        # Features: concat(NMF user factors, NMF course factors).
        # StandardScaler + Ridge regression.
        n_components = params.get('n_components', 15)
        ucm = build_weighted_user_course_matrix()
        nmf = NMF(n_components=n_components, init='nndsvda',
                  random_state=42, max_iter=1000, l1_ratio=0.5)
        W = nmf.fit_transform(ucm.values)
        H = nmf.components_
        X_train, y_train = _build_embedding_dataset(ucm, W, H)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        reg = Ridge(alpha=0.5)
        reg.fit(X_train, y_train)
        joblib.dump((reg, nmf, H, ucm, scaler), os.path.join(MODEL_DIR, 'regression.pkl'))

    elif model_name == models[8]:
        # Classification with Embedding Features
        # Label: 3.0 (completed) → 1, 2.0 (audited) → 0.
        # class_weight='balanced' corrects the ~20:1 class imbalance.
        n_components = params.get('n_components', 15)
        ucm = build_weighted_user_course_matrix()
        nmf = NMF(n_components=n_components, init='nndsvda',
                  random_state=42, max_iter=1000, l1_ratio=0.5)
        W = nmf.fit_transform(ucm.values)
        H = nmf.components_
        X_train, y_raw = _build_embedding_dataset(ucm, W, H)
        # y_raw is normalised to [0,1]; 1.0 = completed (3.0/3.0), 0.5 = audited
        y_train = (y_raw >= 1.0).astype(int)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=500, random_state=42,
                                 class_weight='balanced', C=1.0)
        clf.fit(X_train, y_train)
        joblib.dump((clf, nmf, H, ucm, scaler), os.path.join(MODEL_DIR, 'classification.pkl'))


# ---- Prediction ----

def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users, courses, scores = [], [], []

    for user_id in user_ids:
        ratings_df = load_ratings()
        user_ratings = ratings_df[ratings_df['user'] == user_id]
        enrolled_course_ids = user_ratings['item'].to_list()
        # Weight by rating: completed=1.0, audited=0.5
        rating_weight = {row['item']: (1.0 if row['rating'] >= 3.0 else 0.5)
                         for _, row in user_ratings.iterrows()}

        if model_name == models[0]:
            # ---- Course Similarity (unchanged) ----
            res = course_similarity_recommendations(
                idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for course, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(score)

        elif model_name == models[1]:
            # ---- User Profile (TF-IDF weighted cosine similarity) ----
            # Build TF-IDF course matrix
            threshold = params.get('profile_sim_threshold', 50) / 100.0
            tfidf_matrix, _ = build_tfidf_course_matrix()
            enrolled_in_matrix = [c for c in enrolled_course_ids if c in tfidf_matrix.index]
            if enrolled_in_matrix:
                # Weight each enrolled course vector by its rating weight
                weights = np.array([rating_weight.get(c, 1.0) for c in enrolled_in_matrix])
                course_vecs = tfidf_matrix.loc[enrolled_in_matrix].values
                user_profile = np.average(course_vecs, axis=0, weights=weights)
                # Cosine similarity between user profile and all courses
                sim_scores = cosine_sim(user_profile, tfidf_matrix.values)
                score_series = pd.Series(sim_scores, index=tfidf_matrix.index)
                unenrolled = [c for c in score_series.index if c not in enrolled_course_ids]
                score_series = score_series[unenrolled].sort_values(ascending=False)
                for course, score in score_series.items():
                    if score >= threshold:
                        users.append(user_id)
                        courses.append(course)
                        scores.append(float(score))

        elif model_name == models[2]:
            # ---- Clustering (enrollment rate scoring) ----
            # Score = proportion of cluster members who enrolled in each course.
            # Far more meaningful than mean rating on a 95%-sparse matrix.
            model_path = os.path.join(MODEL_DIR, 'clustering.pkl')
            if not os.path.exists(model_path):
                continue
            kmeans, ucm = joblib.load(model_path)
            user_vec = get_user_vector(enrolled_course_ids, ucm.columns.tolist())
            cluster_label = kmeans.predict([user_vec])[0]
            cluster_mask = kmeans.labels_ == cluster_label
            cluster_courses = ucm.iloc[cluster_mask]
            cluster_size = cluster_mask.sum()
            # Enrollment rate: how many cluster members took each course
            enroll_rate = (cluster_courses > 0).sum(axis=0) / cluster_size
            unenrolled = [c for c in enroll_rate.index if c not in enrolled_course_ids]
            enroll_rate = enroll_rate[unenrolled].sort_values(ascending=False)
            for course, score in enroll_rate.items():
                if score > 0:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(float(score))

        elif model_name == models[3]:
            # ---- Clustering with PCA (enrollment rate scoring) ----
            model_path = os.path.join(MODEL_DIR, 'clustering_pca.pkl')
            if not os.path.exists(model_path):
                continue
            kmeans, pca, scaler, ucm = joblib.load(model_path)
            user_vec = get_user_vector(enrolled_course_ids, ucm.columns.tolist())
            user_scaled = scaler.transform([user_vec])
            user_pca = pca.transform(user_scaled)
            cluster_label = kmeans.predict(user_pca)[0]
            cluster_mask = kmeans.labels_ == cluster_label
            cluster_courses = ucm.iloc[cluster_mask]
            cluster_size = cluster_mask.sum()
            enroll_rate = (cluster_courses > 0).sum(axis=0) / cluster_size
            unenrolled = [c for c in enroll_rate.index if c not in enrolled_course_ids]
            enroll_rate = enroll_rate[unenrolled].sort_values(ascending=False)
            for course, score in enroll_rate.items():
                if score > 0:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(float(score))

        elif model_name == models[4]:
            # ---- KNN (distance-weighted scoring) ----
            # Root cause fix: nearest neighbors may only have the SAME courses
            # as the user (distance=0, single-course users), yielding 0 recommendations.
            # Solution: fetch a large candidate pool (max 100), discard neighbors
            # who have NO unenrolled courses to offer, then score the remainder.
            model_path = os.path.join(MODEL_DIR, 'knn.pkl')
            if not os.path.exists(model_path):
                continue
            knn, ucm = joblib.load(model_path)
            user_vec = get_user_vector(enrolled_course_ids, ucm.columns.tolist())
            user_vec_norm = normalize([user_vec], norm='l2')[0]

            # Fetch a large candidate pool so we survive after filtering useless neighbors
            k_requested = params.get('n_neighbors', 10)
            candidate_pool = min(max(k_requested * 10, 100), len(ucm))
            distances, indices = knn.kneighbors([user_vec_norm], n_neighbors=candidate_pool)
            dists = distances[0]
            idxs = indices[0]

            enrolled_set = set(enrolled_course_ids)
            col_list = ucm.columns.tolist()

            # Keep only neighbors who have at least one unenrolled course to offer
            useful_mask = []
            for idx in idxs:
                neighbor_courses = set(col_list[j] for j in range(len(col_list))
                                       if ucm.iloc[idx, j] > 0)
                has_new = bool(neighbor_courses - enrolled_set)
                useful_mask.append(has_new)
                if sum(useful_mask) >= k_requested:
                    break  # collected enough useful neighbors

            useful_indices = [idxs[i] for i, m in enumerate(useful_mask) if m]
            useful_dists  = [dists[i]  for i, m in enumerate(useful_mask) if m]

            if not useful_indices:
                # Fallback: use all candidates regardless
                useful_indices = list(idxs)
                useful_dists   = list(dists)

            weights = np.maximum(1.0 - np.array(useful_dists), 1e-6)
            neighbor_matrix = ucm.iloc[useful_indices].values
            binary_matrix   = (neighbor_matrix > 0).astype(float)
            weighted_scores = (binary_matrix * weights[:, None]).sum(axis=0) / weights.sum()

            score_series = pd.Series(weighted_scores, index=ucm.columns)
            unenrolled = [c for c in score_series.index if c not in enrolled_set]
            score_series = score_series[unenrolled].sort_values(ascending=False)
            for course, score in score_series.items():
                if score > 0:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(float(score))

        elif model_name == models[5]:
            # ---- NMF (weighted matrix, normalised predictions) ----
            model_path = os.path.join(MODEL_DIR, 'nmf.pkl')
            if not os.path.exists(model_path):
                continue
            nmf, _, H, ucm = joblib.load(model_path)
            # Build user vector with same weighting used in training
            user_vec = np.zeros(len(ucm.columns))
            col_index = {c: i for i, c in enumerate(ucm.columns)}
            for course, w in rating_weight.items():
                if course in col_index:
                    user_vec[col_index[course]] = w
            user_factors = nmf.transform([user_vec])[0]
            predicted = user_factors @ H
            # Normalise scores to [0,1] for interpretability
            if predicted.max() > 0:
                predicted = predicted / predicted.max()
            score_series = pd.Series(predicted, index=ucm.columns)
            unenrolled = [c for c in score_series.index if c not in enrolled_course_ids]
            score_series = score_series[unenrolled].sort_values(ascending=False)
            for course, score in score_series.items():
                if score > 0.01:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(float(score))

        elif model_name == models[6]:
            # ---- Neural Network (hybrid NMF + TF-IDF) ----
            model_path = os.path.join(MODEL_DIR, 'neural_network.pkl')
            if not os.path.exists(model_path):
                continue
            mlp, nmf, H, ucm, svd, course_svd_map, scaler = joblib.load(model_path)
            user_vec = np.zeros(len(ucm.columns))
            col_index = {c: i for i, c in enumerate(ucm.columns)}
            for course, w in rating_weight.items():
                if course in col_index:
                    user_vec[col_index[course]] = w
            user_factors = nmf.transform([user_vec])[0]
            X_pred = np.array([
                np.concatenate([user_factors, H[:, j]])
                for j in range(len(ucm.columns))
            ])
            X_pred = scaler.transform(X_pred)
            predicted = mlp.predict(X_pred)
            score_series = pd.Series(predicted, index=ucm.columns)
            unenrolled = [c for c in score_series.index if c not in enrolled_course_ids]
            score_series = score_series[unenrolled].sort_values(ascending=False)
            for course, score in score_series.items():
                if score > 0:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(float(score))

        elif model_name == models[7]:
            # ---- Regression with Embedding Features ----
            model_path = os.path.join(MODEL_DIR, 'regression.pkl')
            if not os.path.exists(model_path):
                continue
            reg, nmf, H, ucm, scaler = joblib.load(model_path)
            user_vec = np.zeros(len(ucm.columns))
            col_index = {c: i for i, c in enumerate(ucm.columns)}
            for course, w in rating_weight.items():
                if course in col_index:
                    user_vec[col_index[course]] = w
            user_factors = nmf.transform([user_vec])[0]
            X_pred = np.array([
                np.concatenate([user_factors, H[:, j]])
                for j in range(len(ucm.columns))
            ])
            X_pred = scaler.transform(X_pred)
            predicted = reg.predict(X_pred)
            score_series = pd.Series(predicted, index=ucm.columns)
            unenrolled = [c for c in score_series.index if c not in enrolled_course_ids]
            score_series = score_series[unenrolled].sort_values(ascending=False)
            for course, score in score_series.items():
                if score > 0:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(float(score))

        elif model_name == models[8]:
            # ---- Classification with Embedding Features ----
            # Score = P(class=1 | features), i.e. probability the user will complete the course.
            model_path = os.path.join(MODEL_DIR, 'classification.pkl')
            if not os.path.exists(model_path):
                continue
            clf, nmf, H, ucm, scaler = joblib.load(model_path)
            user_vec = np.zeros(len(ucm.columns))
            col_index = {c: i for i, c in enumerate(ucm.columns)}
            for course, w in rating_weight.items():
                if course in col_index:
                    user_vec[col_index[course]] = w
            user_factors = nmf.transform([user_vec])[0]
            X_pred = np.array([
                np.concatenate([user_factors, H[:, j]])
                for j in range(len(ucm.columns))
            ])
            X_pred = scaler.transform(X_pred)
            probs = clf.predict_proba(X_pred)[:, 1]
            score_series = pd.Series(probs, index=ucm.columns)
            unenrolled = [c for c in score_series.index if c not in enrolled_course_ids]
            score_series = score_series[unenrolled].sort_values(ascending=False)
            for course, score in score_series.items():
                if score >= 0.5:
                    users.append(user_id)
                    courses.append(course)
                    scores.append(float(score))

    res_df = pd.DataFrame({'USER': users, 'COURSE_ID': courses, 'SCORE': scores},
                          columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
