"""Encoder for mapping strings to integer IDs."""

from typing import Any


class StringEncoder:
    """
    Encodes strings as integer IDs and decodes them back.
    
    Strings are assigned IDs in the order they are first encountered.
    This encoder is deterministic for a given set of strings.
    """

    def __init__(self) -> None:
        """Initialize an empty encoder."""
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._next_id: int = 0

    def fit(self, names: list[str]) -> None:
        """
        Fit the encoder to a list of strings.
        
        Args:
            names: List of strings to encode
        """
        for name in names:
            if name not in self._name_to_id:
                self._name_to_id[name] = self._next_id
                self._id_to_name[self._next_id] = name
                self._next_id += 1

    def encode(self, names: list[str]) -> list[int | None]:
        """
        Encode strings to integer IDs.
        
        Args:
            names: List of strings to encode
            
        Returns:
            List of integer IDs for the given strings
        """
        return [self._name_to_id.get(name, None) for name in names]

    def encode_known(self, names: list[str]) -> tuple[list[int], list[str]]:
        """
        Encode strings, filtering out unknown values.
        
        Args:
            names: List of strings to encode
            
        Returns:
            Tuple of (encoded_ids, known_names) containing only the strings that were found
        """
        encoded_ids = []
        known_names = []
        
        for name in names:
            encoded_id = self._name_to_id.get(name, None)
            if encoded_id is not None:
                encoded_ids.append(encoded_id)
                known_names.append(name)
        
        return encoded_ids, known_names

    def decode(self, ids: list[int]) -> list[str | None]:
        """
        Decode integer IDs back to strings.
        
        Args:
            ids: List of integer IDs to decode
            
        Returns:
            List of strings for the given IDs
        """
        return [self._id_to_name.get(id_, None) for id_ in ids]

    @property
    def size(self) -> int:
        """Get the number of unique strings in the encoder."""
        return len(self._name_to_id)

    @property
    def names(self) -> list[str]:
        """Get all strings in the encoder (ordered by ID)."""
        return [self._id_to_name[i] for i in range(self.size)]

    def __contains__(self, name: str) -> bool:
        """Check if a string is in the encoder."""
        return name in self._name_to_id

    def to_dict(self) -> dict[str, Any]:
        """
        Convert encoder to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the encoder
        """
        return {
            "name_to_id": self._name_to_id,
            "id_to_name": self._id_to_name,
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StringEncoder":
        """
        Create encoder from a dictionary.
        
        Args:
            data: Dictionary representation from to_dict()
            
        Returns:
            StringEncoder instance
        """
        encoder = cls()
        encoder._name_to_id = data["name_to_id"]
        # Convert string keys back to integers for id_to_name
        encoder._id_to_name = {int(k): v for k, v in data["id_to_name"].items()}
        encoder._next_id = data["next_id"]
        return encoder

