import time
import logging
import os
import sys
import numpy as np
from utils.scraper import Scraper
from utils.predictor import Predictor

# Ensure the logs directory exists
logs_dir = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "app.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def main():
    try:
        scraper = Scraper()
        predictor = Predictor()

        initial_numbers = [
            1.14, 1.56, 2.34, 1.87, 3.45, 1.23, 4.56, 1.98, 2.67, 1.45,
            5.43, 1.76, 2.89, 1.32, 3.78, 1.54, 2.12, 1.89, 4.32, 1.65,
            2.45, 1.78, 3.21, 1.43, 2.98, 1.67, 4.12, 1.34, 2.76, 1.99,
            3.54, 1.88, 2.23, 1.47, 4.87, 1.56, 2.65, 1.92, 3.33, 1.29,
            2.44, 1.73, 4.01, 1.66, 2.88, 1.55, 3.67, 1.41, 2.19, 1.83
        ]
        all_numbers = initial_numbers.copy()
        processed_numbers = set(all_numbers)

        # Initial prediction with 50 numbers
        result = predictor.train_and_predict(all_numbers)
        if isinstance(result, dict):
            logging.info(f"Initial Prediction - Predicted Range: {result['predicted_range'].capitalize()} ({result['range_min']}x - {result['range_max']}x), Probability: {result['probability']:.2f}%")
        else:
            logging.warning(f"Initial prediction failed: {result}")

        # Main loop for continuous scraping and prediction
        while True:
            try:
                new_numbers = scraper.scrape_numbers(processed_numbers)
                if new_numbers:
                    all_numbers.extend(new_numbers)
                    logging.info(f"Total numbers after scrape: {len(all_numbers)}")
                    result = predictor.train_and_predict(all_numbers)
                    if isinstance(result, dict):
                        logging.info(f"Predicted Range: {result['predicted_range'].capitalize()} ({result['range_min']}x - {result['range_max']}x), Probability: {result['probability']:.2f}%")
                    else:
                        logging.warning(f"Prediction failed: {result}")
                else:
                    logging.warning("No new numbers found.")

                time.sleep(scraper.current_interval)

            except KeyboardInterrupt:
                logging.info("Script stopped by user.")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(5)  # Wait before retrying

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        logging.info("Shutting down.")

if __name__ == "__main__":
    main()