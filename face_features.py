
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# =====================================================
# MEDIAPIPE — Compatibility Import
# =====================================================

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except (AttributeError, ImportError):
    # Fallback for newer mediapipe packaging
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
        mp_face_mesh = mp_face_mesh_module
        import mediapipe as mp
    except Exception:
        raise ImportError(
            "MediaPipe face_mesh not available. "
            "Please install: pip install mediapipe==0.10.8"
        )

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



# =====================================================
# LANDMARKS
# =====================================================

LIP_INDICES = [
    61,146,91,181,84,17,314,405,321,375,
    291,308,324,318,402,317,14,87,178,88
]

TOP_LIP = 13
BOTTOM_LIP = 14

LEFT_CORNER = 61
RIGHT_CORNER = 291

CHIN = 152
FOREHEAD = 10

# =====================================================
# MAR
# =====================================================

def mouth_aspect_ratio(face):

    top = face.landmark[TOP_LIP]
    bottom = face.landmark[BOTTOM_LIP]

    left = face.landmark[LEFT_CORNER]
    right = face.landmark[RIGHT_CORNER]

    mouth_height = np.sqrt(
        (top.x-bottom.x)**2 +
        (top.y-bottom.y)**2
    )

    mouth_width = np.sqrt(
        (left.x-right.x)**2 +
        (left.y-right.y)**2
    )

    mar = mouth_height / (mouth_width + 1e-8)

    return mar

# =====================================================
# JAW OPENING
# =====================================================

def jaw_opening(face):

    chin = face.landmark[CHIN]
    forehead = face.landmark[FOREHEAD]

    return abs(chin.y - forehead.y)

# =====================================================
# LIP FEATURE VECTOR
# =====================================================

def lip_feature_vector(face, w, h):

    pts = []

    for idx in LIP_INDICES:

        p = face.landmark[idx]

        x = p.x * w
        y = p.y * h

        pts.append([x, y])

    pts = np.array(pts)

    norm_pts = (
        pts - np.mean(pts)
    ) / (np.std(pts) + 1e-8)

    return norm_pts.flatten(), pts

# =====================================================
# VELOCITY
# =====================================================

def lip_velocity(curr_pts, prev_pts):

    if prev_pts is None:
        return 0

    curr_center = np.mean(curr_pts, axis=0)
    prev_center = np.mean(prev_pts, axis=0)

    return np.linalg.norm(
        curr_center - prev_center
    )

# =====================================================
# ARTICULATION MATRIX
# =====================================================

def articulation_vector(face, w, h):

    mar = mouth_aspect_ratio(face)

    jaw = jaw_opening(face)

    left = face.landmark[LEFT_CORNER]
    right = face.landmark[RIGHT_CORNER]

    width = np.sqrt(
        (left.x-right.x)**2 +
        (left.y-right.y)**2
    )

    return np.array([
        mar,
        jaw,
        width
    ])

# =====================================================
# VIDEO FEATURE EXTRACTION
# =====================================================

def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)

    lip_features = []
    mar_values = []
    jaw_values = []
    velocity_values = []
    articulation_matrix = []

    prev_pts = None

    detected_frames = 0

    preview_frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:

            detected_frames += 1

            face = result.multi_face_landmarks[0]

            lip_feat, pts = lip_feature_vector(
                face,
                w,
                h
            )

            lip_features.append(
                lip_feat
            )

            mar = mouth_aspect_ratio(
                face
            )

            jaw = jaw_opening(
                face
            )

            vel = lip_velocity(
                pts,
                prev_pts
            )

            art_vec = articulation_vector(
                face,
                w,
                h
            )

            mar_values.append(
                mar
            )

            jaw_values.append(
                jaw
            )

            velocity_values.append(
                vel
            )

            articulation_matrix.append(
                art_vec
            )

            prev_pts = pts

            # draw landmarks

            for p in pts.astype(int):

                cv2.circle(
                    frame,
                    tuple(p),
                    2,
                    (0,255,0),
                    -1
                )

        preview_frames.append(
            frame
        )

    cap.release()

    return {
        "lip": np.array(lip_features),
        "mar": np.array(mar_values),
        "jaw": np.array(jaw_values),
        "velocity": np.array(velocity_values),
        "articulation": np.array(
            articulation_matrix
        ),
        "frames": preview_frames,
        "detected": detected_frames
    }

# =====================================================
# DTW SIMILARITY
# =====================================================

def dtw_similarity(a, b):

    if len(a) == 0 or len(b) == 0:
        return 0

    distance, _ = fastdtw(
        a,
        b,
        dist=euclidean
    )

    norm = distance / (
        len(a) + len(b) + 1e-8
    )

    return 1 / (1 + norm)

# =====================================================
# FEATURE COMPARISON
# =====================================================

def compare_features(
    therapist,
    patient
):

    lip_score = dtw_similarity(
        therapist["lip"],
        patient["lip"]
    )

    mar_score = dtw_similarity(
        therapist["mar"].reshape(-1,1),
        patient["mar"].reshape(-1,1)
    )

    jaw_score = dtw_similarity(
        therapist["jaw"].reshape(-1,1),
        patient["jaw"].reshape(-1,1)
    )

    vel_score = dtw_similarity(
        therapist["velocity"].reshape(-1,1),
        patient["velocity"].reshape(-1,1)
    )

    art_score = dtw_similarity(
        therapist["articulation"],
        patient["articulation"]
    )

    return {
        "lip": lip_score,
        "mar": mar_score,
        "jaw": jaw_score,
        "velocity": vel_score,
        "articulation": art_score
    }