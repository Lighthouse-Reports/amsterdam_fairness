import os
import traceback

import aiohttp
from fraude_preventie.api.healthcheck import Healthcheck


class WPIHealthcheck(Healthcheck):
    def __init__(self, scorer):
        super().__init__(check_database_connections=False)
        self.scorer = scorer
        self.register_check("check_daso_api", self._check_daso_api)
        self.register_check("check_model", self._check_model)

    async def _check_model(self):
        """Check if the score function of the model works"""
        applications = [
            "20185071",
        ]
        try:
            result = await self.scorer.score(tuple(applications))
        except Exception:
            return {
                "success": False,
                "model": "Test score failed!",
                "exception": traceback.format_exc(),
            }
        if result["score"].isnull().any():
            return {"success": False, "info": "Null value present in the prediction."}
        return {"success": True, "info": "Model working."}

    @staticmethod
    async def _check_daso_api():
        base_url = f"{os.environ['DATA_API_URL']}/"
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url) as r:
                if r.status != 200:
                    return {
                        "success": False,
                        "DASO API": "API check failed!",
                        "exception": f"The test endpoint responded with status: {r.status} and message: {r.content.read()}",
                    }
                return {"success": True, "info": "DASO API is reachable."}
