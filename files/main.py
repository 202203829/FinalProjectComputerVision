import calibration 
import tracker 
import patron_detection
from time import sleep
if __name__ == "__main__":
    calibration.main()
    patron_detection.main()
    sleep(20)
    tracker.main()