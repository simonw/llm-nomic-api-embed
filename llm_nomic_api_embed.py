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
    # and the combo models
    register(
        NomicCombinedModel(
            model_id="nomic-embed-combined-v1",
            text_model_id="nomic-embed-text-v1",
            vision_model_id="nomic-embed-vision-v1",
        )
    )
    register(
        NomicCombinedModel(
            model_id="nomic-embed-combined-v1.5",
            text_model_id="nomic-embed-text-v1.5",
            vision_model_id="nomic-embed-vision-v1.5",
        )
    )


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


class NomicCombinedModel(EmbeddingModel):
    needs_key = "nomic"
    key_env_var = "NOMIC_API_KEY"
    batch_size = 10
    supports_binary = True
    supports_text = True

    def __init__(self, model_id, text_model_id, vision_model_id):
        self.model_id = model_id
        self.text_model_id = text_model_id
        self.vision_model_id = vision_model_id

    def embed_batch(self, items):
        # Split into text and images
        texts = []
        images = []
        results_by_index = {}
        for i, item in enumerate(items):
            if isinstance(item, str):
                texts.append((i, item))
            else:
                images.append((i, item))
        if texts:
            model = llm.get_embedding_model(self.text_model_id)
            vectors = model.embed_batch([item[1] for item in texts])
            for (index, _), vector in zip(texts, vectors):
                results_by_index[index] = vector
        if images:
            model = llm.get_embedding_model(self.vision_model_id)
            vectors = model.embed_batch([item[1] for item in images])
            for (index, _), vector in zip(images, vectors):
                results_by_index[index] = vector
        keys = results_by_index.keys()
        return [results_by_index[key] for key in sorted(keys)]
