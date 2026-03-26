from common_utils.project_paths import resolve_model_zoo_path


MODEL_PATH_MAP = {
    "qwen-image-edit": str(resolve_model_zoo_path("Qwen/Qwen-Image-Edit", "QWEN_IMAGE_EDIT_PATH")),
    "qwen-image-edit-2509": str(resolve_model_zoo_path("Qwen/Qwen-Image-Edit-2509", "QWEN_IMAGE_EDIT_2509_PATH")),
    "qwen-image-edit-2511": str(resolve_model_zoo_path("Qwen/Qwen-Image-Edit-2511", "QWEN_IMAGE_EDIT_2511_PATH")),
    "step1x_edit1.2": str(resolve_model_zoo_path("stepfun-ai/Step1X-Edit-v1p2", "STEP1X_EDIT_V1P2_PATH")),
    "step1x_edit1.2-preview": str(
        resolve_model_zoo_path("stepfun-ai/Step1X-Edit-v1p2-preview", "STEP1X_EDIT_V1P2_PREVIEW_PATH")
    ),
    "kontext": str(resolve_model_zoo_path("FLUX.1-Kontext-dev", "KONTEXT_MODEL_PATH")),
    "flux.2_klein_9b": str(resolve_model_zoo_path("black-forest-labs/FLUX.2-klein-9B", "FLUX2_KLEIN_9B_PATH")),
    "flux.2_klein_4b": str(resolve_model_zoo_path("black-forest-labs/FLUX.2-klein-4B", "FLUX2_KLEIN_4B_PATH")),
    "longcat_image_edit": str(resolve_model_zoo_path("meituan-longcat/LongCat-Image-Edit", "LONGCAT_IMAGE_EDIT_PATH")),
    "glm_image": str(resolve_model_zoo_path("zai-org/GLM-Image", "GLM_IMAGE_MODEL_PATH")),
    "flux.2_dev": str(resolve_model_zoo_path("black-forest-labs/FLUX.2-dev", "FLUX2_DEV_PATH")),
    "flux.2_dev_turbo": str(resolve_model_zoo_path("fal/FLUX.2-dev-Turbo", "FLUX2_DEV_TURBO_PATH")),
    "FireRed-Image-Edit-1.1": str(
        resolve_model_zoo_path("FireRedTeam/FireRed-Image-Edit-1.1", "FIRERED_IMAGE_EDIT_1P1_PATH")
    ),
}
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

QWEN3_VL_EMBEDDING_MODEL_PATH = str(
    resolve_model_zoo_path("Qwen/Qwen3-VL-Embedding-8B", "QWEN3_VL_EMBEDDING_MODEL_PATH")
)
