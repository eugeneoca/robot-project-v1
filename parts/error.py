
class Error():

    # Error Levels
    low = 1
    mid = 2
    high = 3

    def __init__(self):
        pass

    def report(self, message, errorlevel):
        print("Error({errorlevel}): {message}".format(errorlevel=errorlevel, message=message))

        if errorlevel==self.high:
            exit(1) # Due to fatal error