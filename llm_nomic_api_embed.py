from llm import EmbeddingModel, Model, hookimpl
import llm
import httpx


@hookimpl
def register_embedding_models(register):
    register(
        NomicTextModel("nomic-embed-text-v1", "nomic-embed-text-v1"),
        aliases=("nomic-1",),
    )
    for dimensionality in [None, 512, 256, 128, 64]:
        model_id = "nomic-embed-text-v1.5"
        alias = "nomic-1.5"
        if dimensionality:
            model_id += "-" + str(dimensionality)
            alias += "-" + str(dimensionality)
        register(
            NomicTextModel(
                model_id, "nomic-embed-text-v1.5", dimensionality=dimensionality
            ),
            aliases=(alias,),
        )
    register(NomicImageModel("nomic-embed-vision-v1"))
    register(NomicImageModel("nomic-embed-vision-v1.5"))


class NomicTextModel(EmbeddingModel):
    needs_key = "nomic"
    key_env_var = "NOMIC_API_KEY"
    batch_size = 100

    def __init__(self, model_id, nomic_model_id, dimensionality=None):
        self.model_id = model_id
        self.nomic_model_id = nomic_model_id
        self.dimensionality = dimensionality

    def embed_batch(self, items):
        headers = {
            "Authorization": f"Bearer {self.get_key()}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.nomic_model_id,
            "texts": list(items),
        }
        if self.dimensionality:
            data["dimensionality"] = self.dimensionality
        with httpx.Client() as client:
            response = client.post(
                "https://api-atlas.nomic.ai/v1/embedding/text",
                headers=headers,
                json=data,
            )
        response.raise_for_status()
        return response.json()["embeddings"]


class NomicImageModel(EmbeddingModel):
    needs_key = "nomic"
    key_env_var = "NOMIC_API_KEY"
    batch_size = 10
    supports_binary = True
    supports_text = False

    def __init__(self, model_id):
        self.model_id = model_id

    def embed_batch(self, items):
        # items will be a list of bytes
        headers = {
            "Authorization": f"Bearer {self.get_key()}",
        }
        data = {"model": self.model_id}
        files = [("images", item) for item in items]
        with httpx.Client() as client:
            response = client.post(
                "https://api-atlas.nomic.ai/v1/embedding/image",
                headers=headers,
                data=data,
                files=files,
            )
        response.raise_for_status()
        return response.json()["embeddings"]
