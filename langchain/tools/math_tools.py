from langchain.tools import tool
from langchain_core.tools.base import BaseToolkit

@tool
def add_numbers(inputs:str) -> str :
    """
    Calculate the sum of all the numbers passed in comma separated integer string.
    
    Parameters :
    - inputs (str) : A string with comma separated numbers
    
    Return :
    - (str) : A string with result value after adding all the numbers

    Example input : "1, 2, 6"
    Example output : "9"
    """
    
    result = 0
    inputs = inputs.strip().strip('"').split(",")
    for num in inputs :
        if num.strip().isdigit() :
            result += int(num)
    
    return str(result)

@tool
def subtract_numbers(inputs:str) -> str :
    """
    Calculate the difference of all the numbers passed in comma separated integer string.
    
    Parameters :
    - inputs (str) : A string with comma separated numbers
    
    Return :
    - (str) : A string with result value after subtracting all the numbers from the first number in the list

    Example input : "10, 2, 6"
    Example output : "2"
    """
    
    inputs = inputs.strip().strip('"').split(",")
    nums = []
    for num in inputs :
        if num.strip().isdigit() :
            nums.append(float(num))

    result = nums[0]

    for i in range(1,len(nums)) :
        result -= nums[i]
    
    return str(result)

@tool
def multiply_numbers(inputs:str) -> str :
    """
    Calculate the product of all the numbers passed in comma separated integer string.
    
    Parameters :
    - inputs (str) : A string with comma separated numbers
    
    Return :
    - (str) : A string with result value after multiplying all the numbers in the list

    Example input : "10, 2, 6"
    Example output : "120"
    """
    
    inputs = inputs.strip().strip('"').split(",")
    result = 1
    for num in inputs :
        if num.strip().isdigit() :
            result *= (float(num))

    return str(result)

@tool
def divide(inputs: str) -> str:
    """
    Find the division result of two numbers passed in a string format 'numerator / denominator'

    Parameters :
    inputs (str) - A string with division expression in format - 'numerator / denominator'

    Return :
    (str) - A string with division result value.

    Example Input : "10 / 2"
    Example Output: "5"
    """

    num, den = inputs.strip().strip('"').split("/")

    return str(float(num.strip()) / float(den.strip()))

MATH_TOOLS = [add_numbers, subtract_numbers, multiply_numbers, divide]


## Creating a toolkit class that can be used to organize tools

class MathToolkit(BaseToolkit) :

    def get_tools(self):
        return MATH_TOOLS
