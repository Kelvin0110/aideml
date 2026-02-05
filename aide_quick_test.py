import aide
import logging
from dotenv import load_dotenv
load_dotenv()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    aide_logger = logging.getLogger("aide")
    aide_logger.setLevel(logging.INFO)
    print("Starting experiment...")
    exp = aide.Experiment(
        data_dir="example_tasks/bitcoin_price",  # replace this with your own directory
        goal="Build a time series forecasting model for bitcoin close price.",  # replace with your own goal description
        eval="RMSLE"  # replace with your own evaluation metric
    )
    exp.cfg.agent.code.model = "nvdev/openai/gpt-oss-120b"
    exp.cfg.agent.feedback.model = "nvdev/openai/gpt-oss-120b"

    best_solution = exp.run(steps=2)

    print(f"Best solution has validation metric: {best_solution.valid_metric}")
    print(f"Best solution code: {best_solution.code}")
    print("Experiment finished.")

if __name__ == '__main__':
    main()