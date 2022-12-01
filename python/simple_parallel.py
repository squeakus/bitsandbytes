import concurrent.futures
from itertools import repeat

def main():
    # Use multiprocessing to speed up cpu bound tasks
    images = ["1.jpg", "2.jpg", "3.jpg"]
    product_op_list = repeat(product_op, len(wafer_sets))

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCS) as executor:
        results = executor.map(process_image, images)
    print(results)

def process_image(filename):
    result = 1
    return result
