from sentence_transformers import util, SentenceTransformer
import numpy as np
import torch


class EmbeddingModel:

    def __init__(self, hf_token: str|None=None):

        self.hf_token: str = hf_token
        self.name: str = None

        self.location: str = None
        self.remote: bool = False
        self.commit: str = None

        self.device: torch.device = None
        self.model = None


    def load(self,
        location: str,
        name: str='all-mpnet-base-v2',
        remote: bool=False,
        commit: str=None) -> None:

        VALID_MODELS = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']

        if name not in VALID_MODELS:
            raise ValueError(f'Invalid embedding model "{name}". Supported: {VALID_MODELS}')
        
        self.location = location
        self.name = name
        self.remote = remote
        self.commit = commit

        self.model = SentenceTransformer(
            cache_folder=location,
            model_name_or_path=name,
            local_files_only=not self.remote,
            trust_remote_code=self.remote,
            token=self.hf_token)
        
        return
    

    def embed(self, text: str):

        if not self.model:
            raise ValueError('Must load model before using! (see model.load())')

        embedding = self.model.encode(text, convert_to_tensor=True)

        return embedding
    

    def get_similarity(self,
        v1: np.array,
        v2: np.array,
        method: str='torch_dot') -> float:

        methods = ['cosine', 'torch_dot', 'dot']

        if method not in methods:
            raise ValueError(f'Invalid similarity method "method". Supported: {methods}')
        
        if method == 'dot':
            similarity = self._dot_similarity(v1=v1, v2=v2)
        elif method == 'cosine':
            similarity = self._cosine_similarity(v1=v1, v2=v2)
        elif method == 'torch_dot':
            similarity = self._torch_dot_similarity(v1=v1, v2=v2)

        return similarity
    
    @staticmethod
    def _dot_similarity(v1: np.array, v2: np.array) -> float:

        dot_product = torch.dot(v1, v2)

        norm_vector_1 = torch.sqrt(torch.sum(v1**2))
        norm_vector_2 = torch.sqrt(torch.sum(v2**2))

        result = dot_product / (norm_vector_1 * norm_vector_2)

        return result
    

    @staticmethod
    def _torch_dot_similarity(v1: np.array, v2: np.array) -> float:

        result = float(util.dot_score(v1, v2)[0][0])

        return result
    

if __name__ == '__main__':
    model = EmbeddingModel()
    model.load(location=r'<path to model cache>')
    e1 = model.embed(text='What shape is best?')
    e2 = model.embed(text='Hexagons are the bestagons.')
    similarity = model.get_similarity(v1=e1, v2=e2)

    print(similarity)

    pass