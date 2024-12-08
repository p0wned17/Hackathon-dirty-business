from dependency_injector import containers, providers

from src.services.segmentation_detection import MaskGenerator
from src.services.pipeline_v1 import FirstPipeline
from src.services.pipeline_v2 import SecondPipeline


class AppContainer(containers.DeclarativeContainer):
    """
    Контейнер для зависимостей приложения, использующий dependency_injector.
    """

    config = providers.Configuration()

    segmentation_detector = providers.Singleton(
        MaskGenerator,
        config=config.segmentation_detector,
    )
    
    first_pipeline = providers.Singleton(
        FirstPipeline,
        segmentation_detector=segmentation_detector,
    )
    
    second_pipeline = providers.Singleton(
        SecondPipeline,
        segmentation_detector=segmentation_detector,
    )