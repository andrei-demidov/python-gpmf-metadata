""""Exceptions"""


class MetadataException(Exception):
    """Base exception"""


class MetadataFileException(MetadataException):
    """Exceptions related to the file"""


class MetadataFormatException(MetadataException):
    """Exceptions related to the metadata format"""
