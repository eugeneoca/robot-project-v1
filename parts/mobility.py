from time import sleep

class Mobility():

    def __init__(self):
        self.left_motor_state = False
        self.right_motor_state = False

        # Force means, how much force we want the motor move that relates to speed and torque
        # Negative force will lead to reverse motion
        self.left_motor_force = 0
        self.right_motor_force = 0

    def drive(self, heading=0, force=0):
        # Heading in degrees with respect to North
        # Force will be distributed depends on the heading
        pass
        
    def left_motor_en(self, enable=False):
        # Left motor enable test sequence
        self.left_motor_state = enable
        print("Left motor enabled.")

    def right_motor_en(self, enable=False):
        # Right motor enable test sequence
        self.right_motor_state = enable
        print("Righht motor enabled.")

    def left_motor_setForce(self, force=0):
        pass

    def rigth_motor_setForce(self, force=0):
        pass

    def balance_heading(self):
        # This will balance the force while approaching the correct heading.
        pass

    def initialize(self):
        self.left_motor_en(True)
        self.right_motor_en(True)
        sleep(5)
        print("Mobility system initialized.")

    def process(self):
        pass