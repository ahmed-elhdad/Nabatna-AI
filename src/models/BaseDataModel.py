from src.config import get_settings, Settings



        
class BaseDataModel:
    def __init__(self, db_client: object = None):
        """Simple base class to hold DB client reference for models."""
        self.db_client = db_client

    def close(self):
        try:
            if hasattr(self.db_client, 'close'):
                self.db_client.close()
        except Exception:
            pass