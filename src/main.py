import os
import time
from dnadapt.experiments import toyWdgrl, toyFewShotWdgrl, \
    microTcgaWdgrl, microTcgaFewShotWdgrl, hyperparaMicroTcgaWdgrl


if __name__ == '__main__':
    start_time = time.time()
    microTcgaFewShotWdgrl.main()
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time} (s)')

