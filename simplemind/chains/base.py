from abc import ABC, abstractmethod


class BaseChain(ABC):
    """Abstract base class for implementing chain operations.
    
    A Chain represents a processing step that can be executed on input data
    and should be implemented by concrete classes to define specific behaviors.
    """

    @abstractmethod
    def run(self, input_data: str) -> str:
        """Execute the chain's operation on the input data.
        
        Args:
            input_data: The input string to be processed by the chain.
            
        Returns:
            The processed output string.
            
        Raises:
            ValueError: If the input data is invalid or cannot be processed.
        """
        pass
