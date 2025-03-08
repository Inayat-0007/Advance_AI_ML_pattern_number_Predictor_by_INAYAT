import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os

matplotlib.use('Agg')

# Ensure the logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
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

class Visualizer:
    def __init__(self):
        try:
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            plt.ion()
        except Exception as e:
            logging.error(f"Error initializing visualizer: {e}")
            raise

    def plot_history(self, numbers, predictions):
        try:
            self.ax.clear()

            self.ax.plot(numbers, label='Numbers', marker='o', color='blue')

            if len(numbers) >= 10:  # Match window_size from predictor
                moving_avg = pd.Series(numbers).rolling(window=10).mean()
                self.ax.plot(moving_avg, label='Moving Average (10)', linestyle='--', color='orange')

            self.ax.axhline(y=1.99, color='green', linestyle='--', label='Blue Range (1.0-1.99)')
            self.ax.axhline(y=9.99, color='purple', linestyle='--', label='Purple Range (2.0-9.99)')
            self.ax.axhline(y=200.0, color='pink', linestyle='--', label='Pink Range (10.0-200.0)')

            if predictions and numbers:
                last_pred = predictions[-1]
                last_num = numbers[-1]
                self.ax.annotate(last_pred, xy=(len(numbers)-1, last_num),
                                xytext=(len(numbers)-1, last_num+10),
                                arrowprops=dict(facecolor='black', shrink=0.05))

            self.ax.set_xlabel('Round')
            self.ax.set_ylabel('Number (x)')
            self.ax.set_yscale('log')
            self.ax.set_title('Aviator Game Number History and Predicted Ranges')
            self.ax.legend()
            self.ax.grid(True)

            plt.draw()
            plt.pause(0.1)

        except Exception as e:
            logging.error(f"Error during visualization: {e}")