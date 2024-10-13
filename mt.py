"""
Mersenne Twister 64-bit Implementation

This module provides a Python implementation of the Mersenne Twister algorithm
for generating pseudo-random numbers. It is a 64-bit version of the algorithm.

Classes:
    MT: The main Mersenne Twister class providing various random number generation methods.

Exceptions:
    SeedingError: Raised when the generator is not seeded before generating random numbers.
    GeneratorError: Raised when the generator fails generator check.

Functions:
    random: Generates a random float number in the range [0.0, 1.0).
    uniform(a, b): Generates a random float number in the range [a, b).
    randint(start, end): Generates a random integer number in the range [start, end].
    choice(seq): Returns a random element from the given list.
    randrange(start, end=None, step=1): Generates a random integer from range(start, end, step).
    sample(seq, k, unique=True): Returns a list with k unique (or not if unique=False) random elements from the given list.
    shuffle(seq, state=None): Returns a shuffled version of the given list.
    setstate(state): Sets the internal state of the random number generator.

Note: The module exports the above functions directly, so they can be accessed as part of the module itself.
"""


import time
import math
from typing import Any


class SeedingError(Exception):
    """
    Exception raised when the generator is improperly seeded.

    This exception is raised when the Mersenne Twister 64-bit PRNG is used without being properly seeded first.

    Attributes:
        - msg: A message indicating that the generator was never seeded. Developers are encouraged to contact the developer
               responsible for the code to address the seeding issue.
    """

    def __init__(self):
        self.msg = "Generator was never seeded, contact developer!"

    def __str__(self):
        return self.msg


class GeneratorError(Exception):
    """
    Exception raised when the generator encounters an unknown failure.

    This exception is raised when the Mersenne Twister 64-bit PRNG encounters an unknown failure during operation.
    The failure could be due to faulty parameters or potential corruption of the generator code.

    Attributes:
        - msg: A message indicating that the generator failed for an unknown reason. Developers may need to investigate
               and rectify issues related to parameters or the integrity of the generator's implementation.
    """

    def __init__(self):
        self.msg = "Genrator failed for unknown reason. Could be faulty parameters or corruption of the generator code!"

    def __str__(self):
        return self.msg



# List of symbols to be exported when using 'from module import *'
__all__ = [
    "choice",
    "randint",
    "random",
    "randrange",
    "sample",
    "setstate",
    "getstate",
    "shuffle",
    "uniform",
    "permutation",
    "normal"
]


NV_MAGICCONST = 4 * math.exp(-0.5) / math.sqrt(2.0)
LOG4 = math.log(4.0)
SG_MAGICCONST = 1.0 + math.log(4.5)
BPF = 53        # Number of bits in a float
RECIP_BPF = 2 ** -BPF


class MT(object):
    
    """
    Mersenne Twister 64-bit Pseudo-Random Number Generator.
    
    This class implements a Mersenne Twister PRNG with a 64-bit word size,
    providing methods for generating random numbers and performing various
    randomization operations.
    
    Attributes:
        f, w, n, m, r, a, u, d, s, b, t, c, l (int): Constants used in the Mersenne Twister algorithm.
        index (int): Current position in the state vector.
        LM, UM (int): Bitmasks to extract lower and upper bits from a 64-bit word.
        MT (list): Internal state vector for the generator.
    
    Methods:
        twist(): Perform the twist operation to update the internal state.
        computation(width=1000, height=1000): Perform a computation that wastes time,
            used in get_computation_seed() to generate random numbers for seeding.
        get_computation_seed(n=8): Generate a seed using the computation method.
        update(): Update the internal state of the generator.
        _randint(): Generate a random 64-bit integer.
        randrange(start, end=None, step=1): Generate a random integer within a specified range with step.
        randint(start, end=None): Generate a random integer within a specified range [start, end].
        choice(seq): Choose a random element from a list.
        random(): Generate a random floating-point number in the range [0.0, 1.0).
        sample(seq, k, unique=True): Randomly sample unique elements from a list.
        shuffle(seq, state=None): Shuffle a list and return a new shuffled list.
        uniform(a, b): Generate a random floating-point number N such that a <= N <= b.
        setstate(state): Set the internal state of the generator.
    """

    def __init__(self):
        """
        Initializes the Mersenne Twister PRNG with default constants and internal state.
        
        This method sets up the necessary constants and internal state for the Mersenne Twister 
        algorithm, including the coefficients, bitmasks, and initial state vector.
        
        Parameters:
            None
        
        Returns:
            None
        """
        

        # Initialize constants used in the Mersenne Twister algorithm
        self.f = 6364136223846793005
        self.w = 64
        self.n = 312
        self.m = 156
        self.r = 31
        self.a = 0xb5026f5aa96619e9
        self.u = 29
        self.d = 0x5555555555555555
        self.s = 17
        self.b = 0x71d67fffeda60000
        self.t = 37
        self.c = 0xfff7eee000000000
        self.l = 43

        # Initialize index and bitmasks for generating random numbers
        self.index = self.n + 1
        self.LM = (1 << self.r) - 1 
        self.UM = (~self.LM) & ((1 << self.w) - 1)

        # Initialize the internal state vector with zeros and updates it
        self.MT = [0] * self.n
        self.update()

        # Initialize constants for functions
        self.MV = 1 << self.w # Max Value


    def update(self):
        """
        Update the internal state of the Mersenne Twister 64-bit PRNG.

        This method refreshes the internal state of the generator using a two-step process: seeding and recurrence update.

        Seeding:
        1. Decrement the 'index' attribute to ensure that the generator will be considered as unseeded.
        2. Generate a new seed using the 'get_computation_seed()' method, which performs a computation to obtain a seed value.
        3. Set the first element (MT[0]) of the state vector to the new seed.

        Recurrence Update:
        1. For each element from the second element (MT[1]) to the last element (MT[n-1]) of the state vector:
            a. Compute the next state using the Mersenne Twister recurrence formula.
            b. Update the element by applying bitwise operations and arithmetic operations.
        2. The recurrence formula involves XORing the current and shifted state values with constants and applying masks.

        After the update, the generator's internal state is refreshed and ready to generate new random numbers.

        Note:
        - The 'n' attribute represents the size of the state vector.
        - The 'w' attribute represents the word size in bits.
        - The 'f' attribute represents a constant factor.
        - The 'index' attribute is decremented to ensure unseeded state.
        """

        # Decrement the index to signal unseeded state
        self.index -= 1

        # Seed the first element of the state vector
        self.MT[0] = int(str(time.time()).split('.')[1])

        # Recurrence update for other elements in the state vector
        for i in range(1, self.n):
            # Compute the next state using the recurrence formula
            next_state = (self.f * (self.MT[i - 1] ^ (self.MT[i - 1] >> (self.w - 2)))) + i

            # Apply bitwise operations and arithmetic operations to update the element
            self.MT[i] = int(((1 << self.w) - 1) & next_state)


    def twist(self):

        """
        Perform the twist operation to update the internal state of the Mersenne Twister 64-bit PRNG.

        This method updates the internal state of the generator by applying bitwise operations to the current state vector.

        The twist operation involves the following steps for each element in the state vector:
            1. Retrieve the current element MT[i] and the next element MT[(i+1) % n] from the state vector.
            2. Apply bitwise AND operation between MT[i] and a bitmask UM to obtain the lower bits of MT[i].
            3. Apply bitwise AND operation between MT[(i+1) % n] and a bitmask LM to obtain the higher bits of MT[(i+1) % n].
            4. Add the lower bits and higher bits obtained in steps 2 and 3, respectively, to obtain the variable x.
            5. Right-shift x by 1 bit to get xA.
            6. If x is an odd number (i.e., x % 2 != 0), XOR xA with the constant 'a'.
            7. Update MT[i] by XORing the value at position (i + m) % n in the state vector with xA.

        After performing the twist operation for all elements in the state vector, the 'index' attribute is reset to 0,
        indicating that the generator is ready to produce the next random number.

        This method is typically called when the generator's state vector is exhausted, and it needs to be refreshed to
        continue generating random numbers.

        Parameters:
            None

        Returns:
            None

        Note:
            - The 'UM' and 'LM' attributes represent bitmasks to extract the lower and upper bits from a 64-bit word.
            - The 'n' attribute represents the size of the state vector.
            - The 'm' attribute is used in the twist operation to calculate the next index for updating the state vector.
        """

        # Perform the twist operation for each element in the state vector
        for i in range(self.n):
            # Retrieve and add current element and next element from the state vector
            x = (self.MT[i] & self.UM) + (self.MT[(i+1) % self.n] & self.LM)

            # Compute xA by right-shifting x by 1 bit
            xA = x >> 1

            # If x is an odd number, XOR xA with the constant 'a'
            if (x % 2) != 0:
                xA = xA ^ self.a
            
            # Update the state vector element at position i
            self.MT[i] = self.MT[(i + self.m) % self.n] ^ xA

        # Reset the index to 0, indicating the state vector is ready for the next random number generation
        self.index = 0


    def _randint(self) -> int:
        """
        Generate a random 64-bit integer using the Mersenne Twister 64-bit PRNG.

        This method produces a random integer by applying bitwise operations to the current state vector element and then
        updating the internal state for subsequent random number generation.

        The steps involved in generating the random integer are as follows:
            1. Check if the state vector needs to be refreshed (index >= n), and perform the twist operation if necessary.
            2. Retrieve the current state vector element at the current index.
            3. Apply bitwise operations to the current element to introduce randomness:
                - Right-shift number by 'u' bits, then perform bitwise AND with 'd', then perform bitwise XOR with itself.
                - Left-shift number by 's' bits, then perform bitwise AND with 'b', then perform bitwise XOR with itself.
                - Left-shift number by 't' bits, then perform bitwise AND with 'c', then perform bitwise XOR with itself.
                - Right-shift number by 'l' bits, then perform bitwise XOR with itself.
            4. Update the index for the next random number generation.
            5. Return the generated random integer after performing bitwise AND with 64-bit bitmask.

        Parameters:
            None

        Returns:
            int: A random 64-bit integer.

        Note:
            - The attributes 'u', 'd', 's', 'b', 't', and 'l' represent constants used in the bitwise operations.
            - The 'n' attribute represents the size of the state vector.

        Example usage:
        ```
        random_integer = mt._randint()  # Generate a random 64-bit integer.
        ```
        """

        # Check if the state vector needs to be refreshed
        if self.index >= self.n:
            if self.index > self.n:
                raise SeedingError()
            self.twist()

        # Retrieve the current state vector element
        number = self.MT[self.index]

        # Apply bitwise operations to introduce randomness
        number = number ^ ((number >> self.u) & self.d)
        number = number ^ ((number << self.s) & self.b)
        number = number ^ ((number << self.t) & self.c)
        number = number ^ (number >> self.l)
        
        # Update the index for the next random number generation
        self.index += 1

        # Return the generated random integer after masking with a 64-bit bitmask
        return int(((1 << self.w) - 1) & number)
    
    
    def randrange(self, start: int, end: int = None, step: int = 1) -> int:
        """
        Return a randomly selected integer from a range with optional step value.

        This method generates a random integer from the specified range [start, end) using the internal
        Mersenne Twister 64-bit pseudo-random number generator.

        Args:
            start (int): The starting value of the range (inclusive).
            end (int, optional): The ending value of the range (exclusive). Defaults to None.
            step (int, optional): The step size between values in the range. Defaults to 1.

        Returns:
            int: A random integer from the specified range.

        Notes:
            - If end is not provided, start is treated as the upper bound and 0 is used as the lower bound.
            - The range is half-open, meaning it includes the start value but excludes the end value.

        Example usage:
        ```
        random_value = mt.randrange(10)          # Generate a random integer in the range [0, 10).
        random_range = mt.randrange(5, 15)      # Generate a random integer in the range [5, 15).
        random_stepped = mt.randrange(0, 20, 3) # Generate a random integer in the range [0, 20) with a step of 3.
        ```
        """

        # If 'end' is not provided, adjust start and end values
        if end is None:
            end = start
            start = 0

        # Calculate the number of steps within the range
        num_steps = int((end - start) / step) + 1

        # Generate a random index within the number of steps and apply the step value
        random_index = self._randint() % num_steps
        random_value = random_index * step + start

        return int(random_value)
    

    def randint(self, start: int, end: int = None) -> int:
        """
        Generate a random integer within a specified range [start, end] (inclusive).

        This method generates a random integer from the specified range [start, end]. If only one argument 'start' is provided,
        the range is assumed to be [0, start] (inclusive). If 'end' is not provided, it is set to 'start', and 'start' is
        set to 0, effectively generating a random integer in the range [0, end].

        Args:
            start (int): The lower bound of the range (inclusive) if 'end' is specified, or the upper bound (inclusive) if 'end' is not specified.
            end (int, optional): The upper bound of the range (inclusive). Defaults to None.

        Returns:
            int: A randomly generated integer within the specified range [start, end].

        Notes:
            - If 'end' is not provided, the range becomes [0, start].

        Example usage:
        ```
        random_int = mt.randint(1, 10)       # Generate a random integer between 1 and 10 (inclusive).
        random_int_default = mt.randint(5) # Generate a random integer between 0 and 5 (inclusive).
        ```
        """

        # Adjust arguments if 'end' is not provided
        if end is None:
            end = start
            start = 0

        # Delegate to the randrange method to generate a random integer within the specified range
        return self.randrange(start, end)


    def choice(self, seq: list) -> Any:
        """
        Randomly select and return an element from a given list.

        This method chooses a random element from the provided list 'seq' using the internal Mersenne Twister PRNG.
        The selection is performed by generating a random index within the range of the list length and returning the
        element at that index.

        Args:
            seq (list): The list from which to choose a random element.

        Returns:
            item: A randomly selected element from the input list 'seq'.

        Notes:
            - The input 'seq' can be any iterable object, such as a list, tuple, or string.
            - If the input 'seq' is empty, a ValueError will be raised.

        Example usage:
        ```
        colors = ["red", "green", "blue", "yellow"]
        chosen_color = mt.choice(colors)
        print("Randomly chosen color:", chosen_color)
        ```
        """

        # Generate a random index using the PRNG and select the corresponding element from the list
        index = int(self._randint() % len(seq))
        return seq[index]


    def random(self) -> float:
        """
        Generate a random floating-point number in the range [0.0, 1.0] with uniform distribution using the Mersenne Twister 64-bit PRNG.

        This method produces a random floating-point number by dividing the result of the _randint() method by the
        maximum possible value 2**64 (an attribute of the Mersenne Twister class). The result is scaled to the range
        [0.0, 1.0), making it suitable for various randomization tasks.

        Returns:
            float: A random floating-point number in the range [0.0, 1.0).

        Notes:
            - The upper bound (1.0) is excluded from the range.

        Example usage:
        ```
        random_value = mt.random()  # Generate a random floating-point number between 0 and 1 (excluding 1).
        ```
        """

        # Generate a random 64-bit integer and divide it by the maximum possible value plus one - 2**64
        return self._randint() / self.MV


    def sample(self, seq: list, k: int, unique: bool = True) -> list[Any]:
        """
        Randomly sample elements from a list.

        This method selects 'k' random elements from the given list 'seq'. The sampled elements are returned as a list.

        Parameters:
            seq (iterable): The list from which elements will be sampled.
            k (int): The number of elements to be sampled.
            unique (bool, optional): If True, sampled elements are unique (without replacement). If False, elements can be duplicated. Default is True.

        Returns:
            list: A list containing the randomly sampled elements.

        Raises:
            ValueError: If unique is True and k is larger than the length of the list.

        Notes:
            - When unique is True, the method ensures that sampled elements are unique by rejecting duplicates.
            - The 'seq' parameter can be any iterable, such as a list, tuple, or string.
            - If unique is False, the method may return a list with duplicate elements.

        Example usage:
        ```
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Sample 3 unique elements from the list
        unique_sample = mt.sample(numbers, k=3)

        # Sample 4 elements from the list with possible duplicates
        non_unique_sample = mt.sample(numbers, k=4, unique=False)
        ```
        """
        seq_ = seq.copy()
        idxs = []

        # If unique sampling is requested
        if unique:

            # Check if k is valid given the length of the list
            if len(seq) < k:
                raise ValueError("If unique is True, k must be smaller or equal to the length of list")

            # Sample unique indices until 'k' unique indices are obtained
            while len(idxs) < k:
                idx = self._randint() % len(seq_)
                idxs.append(idx)
                seq_.pop(idx)

            # Return the sampled elements using the unique indices
            return [seq[idxs[i]] for i in range(k)]
        
        # If non-unique (with replacement) sampling is requested
        return [seq[self._randint() % len(seq)] for i in range(k)]


    def shuffle(self, seq: list, state: int = None) -> list[Any]:
        """
        Shuffle the elements of a list using the Mersenne Twister 64-bit PRNG.

        This method shuffles the elements of the given list using the Fisher-Yates shuffle algorithm, which is
        enhanced by the Mersenne Twister PRNG for generating random indices.

        Args:
        - seq: The list to be shuffled.
        - state (optional): An optional state to set before shuffling. If provided, the generator's internal state will be
        temporarily updated to the specified state for shuffling and restored afterward.

        Returns:
        - None: Shuffles the list in place and returns None.

        Note:
        - The original list remains unchanged. The returned shuffled list contains the same elements as the input
        list but in a random order.

        Example usage:
        ```
        original_list = [1, 2, 3, 4, 5]
        shuffled_list = mt.shuffle(original_list)
        print(shuffled_list)  # Output may vary: [3, 1, 5, 4, 2] or any other random order
        ```
        """
        if state is not None:
            # Save the current state and set the generator to the provided state (if any)
            state_save = self.MT[self.index]
            # index_save = self.index
            self.setstate(state)

        # Shuffle the list in place using the Fisher-Yates shuffle algorithm
        for i in range(len(seq) - 1, 0, -1):
            j = self._randint() % (i + 1)
            seq[i], seq[j] = seq[j], seq[i]

        if state is not None:
            # Restore the original state if a state was provided
            self.setstate(state_save)


    def permutation(self, seq: list, state: int = None) -> list[Any]:
        if state is not None:
            # Save the current state and set the generator to the provided state (if any)
            state_save = self.MT[self.index]
            # index_save = self.index
            self.setstate(state)

        # Create a copy of the input sequence
        copy_seq = seq.copy()

        # Shuffle the copy list in place using the Fisher-Yates shuffle algorithm
        for i in range(len(copy_seq) - 1, 0, -1):
            j = self._randint() % (i + 1)
            copy_seq[i], copy_seq[j] = copy_seq[j], copy_seq[i]

        if state is not None:
            # Restore the original state if a state was provided
            self.setstate(state_save)

        return copy_seq


    def uniform(self, start: float, end: float) -> float:
        """
        Generate a random floating-point number within the specified range [start, end].

        This method generates a random floating-point number N such that 'start' is less than or equal to N and N is less than
        or equal to 'end'. The generated number is uniformly distributed over the range [start, end].

        Parameters:
        - start (float): The lower bound of the range.
        - end (float): The upper bound of the range.

        Returns:
        float: A random floating-point number within the range [start, end].

        Example usage:
        ```
        random_value = mt.uniform(0.0, 1.0)  # Generate a random number between 0.0 and 1.0 (inclusive).
        ```
        """

        # Generate a random floating-point number in the range [0.0, 1.0]
        random_value = self.random()

        # Scale and shift the random number to the desired range [start, end]
        scaled_value = start + (end - start) * random_value

        return scaled_value
    

    def normal(self, mean: float = 0, sigma: float = 1) -> float:
        """
        Generates a random floating-point number following a normal distribution.

        Parameters:
        - mean (float): The mean of the normal distribution. Defaults to 0.
        - sigma (float): The standard deviation of the normal distribution. Defaults to 1.

        Returns:
        float: A random floating-point number following a normal distribution with the specified mean and standard deviation.
        """
        
        while True:
            u1 = self.random()
            u2 = 1.0 - self.random()
            z = NV_MAGICCONST * (u1 - 0.5) / u2
            zz = z * z / 4.0
            if zz <= -math.log(u2):
                break

        return mean + z * sigma


    def setstate(self, state: int):
        """
        Set the internal state of the Mersenne Twister 64-bit PRNG using a given seed value.

        This method sets the internal state of the generator to a new state determined by the provided seed value. The
        generator's state vector is initialized using the seed value, and subsequent state elements are generated based on
        the Mersenne Twister algorithm.

        Args:
        - state: An integer value used as the seed to initialize the generator's state.

        Example usage:
        ```
        seed_value = 12345
        mt.setstate(seed_value)  # Set the generator's internal state using the provided seed.
        random_number = mt.random()  # Generate a random number using the new state.
        ```

        Important: Changing the internal state will affect the list of generated random numbers. The same seed will
        always produce the same list of random numbers.
        """
        self.MT[0] = state  # Set the initial seed value

        # Generate 311 following seeds from the original seed
        for i in range(1, self.n):
            # Generate the next state element using the Mersenne Twister algorithm
            self.MT[i] = int(((1 << self.w) - 1) & (self.f * (self.MT[i - 1] ^ (self.MT[i - 1] >> (self.w - 2)))) + i)
        
        self.index = 0


    def getstate(self) -> int:
        """
    	Retrieves the current state of the Mersenne Twister generator.

    	Returns:
    	int: The current state of the generator.
    	"""

        # Retrieve the current state of the generator
        state = self.MT[self.index]

        # Set current state to the first element of the MT vector
        self.setstate(state)
        
        return state


def check_generator():
    """
    Verify the correctness of the Mersenne Twister generator implementation by performing a series of tests.

    This function aims to validate the accuracy and consistency of the Mersenne Twister generator's randomization
    operations. It creates an instance of the MT class, applies various randomization methods with predetermined
    inputs, and compares the results with expected outcomes. If any of the generated results do not match the expected
    values, a GeneratorError is raised, indicating a potential issue in the generator implementation.

    Note:
    - This function is intended for testing and debugging purposes to ensure the proper functioning of the generator.

    Raises:
    - GeneratorError: Raised if the generated results do not match the expected outcomes, indicating a potential issue
                        with the Mersenne Twister generator implementation.
    """

    # Create an instance of the Mersenne Twister generator
    check_mt = MT()

    # Set a specific state for testing
    check_mt.setstate(2)

    # Create a list of integers from 0 to 9
    test_list = [i for i in range(10)]

    # Generate test results using the generator's randomization methods
    uni_ = check_mt.uniform(5, 10)
    sample_ = check_mt.sample(test_list, 3)
    choice_ = check_mt.choice(test_list)
    random_ = check_mt.random()
    randint_ = check_mt.randint(0, 10)
    randrange_ = check_mt.randrange(0, 20, 6)
    check_mt.shuffle(test_list)

    # Expected results for comparison
    #[5, 1, 4, 3, 9, 6, 8, 0, 2, 7],  shuffled_list_,
    true_results = ([9, 0, 6, 3, 8, 4, 2, 7, 5, 1], 5.009765699505878, [9, 3, 6], 9, 0.5515543951767912, 3, 12)

    # Compare generated results with expected outcome
    if (test_list, uni_, sample_, choice_, random_, randint_, randrange_) != true_results:
        raise GeneratorError()


# Verify the correctness of the Mersenne Twister generator implementation.
check_generator()

# Instantiate the Mersenne Twister generator for module-level use.
_inst = MT()

# Create module-level aliases for selected generator methods and attributes.
random = _inst.random
uniform = _inst.uniform
randint = _inst.randint
choice = _inst.choice
randrange = _inst.randrange
sample = _inst.sample
shuffle = _inst.shuffle
setstate = _inst.setstate
getstate = _inst.getstate
permutation = _inst.permutation
normal = _inst.normal


"""
The code block above serves two main purposes:
1. It first calls the `check_generator()` function to ensure the accuracy and reliability of the Mersenne Twister
   generator implementation by running a battery of predefined tests.

2. It then creates a module-level instance of the Mersenne Twister generator, named `_inst`, which can be used to access
   its various randomization methods and attributes. Additionally, it creates module-level aliases for selected
   generator methods and attributes, allowing easier and more concise access to these functionalities throughout the
   entire module.

Note:
- This code block is typically placed at the end of a module to perform initial verification and provide convenient access
  to the Mersenne Twister generator's capabilities.
- The module-level aliases allow you to use these generator methods and attributes without explicitly creating an
  instance of the generator every time you need to generate random numbers or perform randomization operations.
"""
