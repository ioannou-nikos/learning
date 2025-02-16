import time
import random

def generate_mock_data():
    while True:
        record = {
            'id': random.randint(1,1000),
            'value': random.random() * 100
        }
        yield record
        time.sleep(0.5) # simulate data arriving every 0.5 seconds

def transform_data(record):
    transformed_record = {
        'id': record['id'],
        'value': record['value'],
        'transformed_value': record['value'] * 1.1 # Example transformation
    }
    return transformed_record

def load_data(record):
    print(f"Loading record into the database: {record}")

def process_stream(run_time_seconds=10):
    start_time = time.time()
    for record in generate_mock_data():
        transformed_record = transform_data(record)
        load_data(transformed_record)
        # Check if the run time has exceeded the limit
        if time.time() - start_time > run_time_seconds:
            print("Time limit reached. Terminating the stream processing.")
            break


if __name__ == "__main__":
    process_stream()