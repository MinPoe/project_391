"""
PID Controller module.

This module implements the PID controller that was described in class with tunable parameters for Kp, Ki, Kd.
"""

class PID:
    """
    PID controller with time-based integration and differentiation.
    """

    def __init__(self, K_p, K_i, K_d) -> None:
        """
        Initialization function for the PID controller.
        
        Args:
            K_p: Proportional gain = how strongly to react to current error
            K_i: Integral gain = how strongly to react to accumulated error
            K_d: Derivative gain = how strongly to react to change in error
        """

        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        self.prev_err = 0
        self.int_acc = 0
        self.prev_time = None

    def pid_err(self, curr_err, current_time) -> float:
        """
        Calculate PID control output based on current error.

        Computes steering correction by combining P, I, D terms. 
        Uses actual time between callbacks (dt) for integration and differentiation.
        
        Args:
            curr_err: Current error measurement (distance from desired position)
            current_time: Current timestamp in seconds from ROS clock
            
        Returns:
            Control output (steering angle correction)
        """

        # calculate time between callbacks
        if self.prev_time is None: 
            t_step = 0.01 # first iteration placeholder variable
        else: 
            t_step = current_time - self.prev_time # recalculate every iteration after first
            if t_step <= 0.0:
                t_step = 1e-6
        
        # Proportional Error
        p_err = self.K_p * curr_err

        # Integral (Accumulated) Error
        self.int_acc += curr_err * t_step 
        self.int_acc = max(min(self.int_acc, 100.0), -100.0) # lower + upper bound for accumulation (edge cases where integral might grow too big)
        i_err = self.K_i * self.int_acc

        # Derivative Error 
        d_err = self.K_d * (curr_err - self.prev_err) / t_step 

        # Next iteration setup
        self.prev_err = curr_err 
        self.prev_time = current_time 

        return p_err + i_err + d_err
