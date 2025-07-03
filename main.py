from src.text_summarizer.logging import logger
from src.text_summarizer.pipeline.stage1_data_ingestion import DataIngestionTrainingPipeline
from src.text_summarizer.pipeline.stage2_data_transformation import DataTransformationTrainingPipeline
# from src.text_summarizer.pipeline.stage3_model_tariner import ModelTrainerTrainingPipeline
from src.text_summarizer.pipeline.stage4_model_evaluation import ModelEvaluationTrainingPipeline
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation=DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

# STAGE_NAME = "Model Trainer stage"
# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     model_trainer=ModelTrainerTrainingPipeline()
#     model_trainer.initiate_model_trainer()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
# except Exception as e:
#     logger.exception(e)   
#     raise e

STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation=ModelEvaluationTrainingPipeline()
    model_evaluation.initiate_model_evaluation()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e