import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.containers import AppContainer
from src.routes import dirty as dirty_routes
from src.routes.routers import router as app_router


def create_app() -> FastAPI:
    container = AppContainer()
    cfg = OmegaConf.load("config/config.yml")
    container.config.from_dict(cfg)
    container.wire([dirty_routes])

    app = FastAPI()
    app.include_router(app_router, prefix="/dirty_segment", tags=["dirty_segment"])
    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, port=8888, host="0.0.0.0")