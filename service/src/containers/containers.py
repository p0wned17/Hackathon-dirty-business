from dependency_injector import containers, providers

from src.services.segmentation_detection import MaskGenerator



class AppContainer(containers.DeclarativeContainer):
    """
    Контейнер для зависимостей приложения, использующий dependency_injector.
    """

    config = providers.Configuration()

    segmentation_detector = providers.Singleton(
        MaskGenerator,
        config=config.segmentation_detector,
    )