"""GPMF header metadata extractor for GoPro MP4, LRV, 360 and JPG files"""

import os
import sys

from gpmf_metadata.exceptions import MetadataFileException, MetadataFormatException


class Metadata:
    """Find, extract, and parse GPMF header metadata"""

    __filename : str
    __header : bytes
    __is_photo : bool
    __gpmf_offset : int
    __gpmf_size : int
    __metadata : list[list[str, str | int | float | list | None]]
    __iter_counter : int

    def __init__(self, filename : str) -> None:
        """Initialize the object and process the file"""
        if not os.path.exists(filename):
            raise MetadataFileException("The file doesn't exist.")
        self.__filename = filename
        self.__header = bytes()
        self.__is_photo = False
        self.__gpmf_offset = 0
        self.__gpmf_size = 0
        self.__metadata = []
        self.__iter_counter = 0
        self.__parse_metadata()

    @property
    def is_photo(self) -> bool:
        """Return True if the processed file is a photo"""
        return self.__is_photo

    @property
    def is_video(self) -> bool:
        """Return True if the processed file is a video"""
        return not self.__is_photo

    @property
    def metadata(self) -> list:
        """Return the metadata"""
        return self.__metadata

    def __get_video_header_offset(self, data : bytes) -> int:
        """Get the header offset in a video file"""
        offset = 0
        for i in range(0, 60):
            if data[i] == 109 and data[i + 1] == 100 and data[i + 2] == 97 and data[i + 3] == 116:
                if data[i - 4] == 0 and data[i - 3] == 0 and data[i - 2] == 0 and data[i - 1] == 1:
                    offset = data[i + 7] * 4294967296
                    offset += data[i + 8] * 16777216
                    offset += (data[i + 9] << 16) + (data[i + 10] << 8) + \
                        (data[i + 11] << 0) + i - 4
                else:
                    offset = data[i - 4] * 16777216
                    offset += (data[i - 3] << 16) + (data[i - 2] << 8) + (data[i - 1] << 0) + i - 4
        return offset

    def __get_photo_header_offset(self, data : bytes) -> list[int, int]:
        """Get the header offset in a photo file"""
        offset = 0
        size = 0
        i = 0
        while True:
            while data[i] == 0xff and (data[i + 1] < 0xe0 or data[i + 1] > 0xef):
                i += 2
            if data[i] == 0xff and data[i + 1] >= 0xe1 and data[i + 1] <= 0xef:
                if data[i + 1] == 0xe6:
                    if data[i + 4] == 0x47 and data[i + 5] == 0x6F and data[i + 6] == 0x50 and \
                        data[i + 7] == 0x72 and data[i + 8] == 0x6F:
                        # Valid JPEG
                        offset = i + 10
                        size = (data[i + 2] << 8) + (data[i + 1] << 0)
                    break
                i += (data[i + 2] << 8) + (data[i + 3]) + 2
            else:
                # Invalid JPEG
                break
        return offset, size

    def __get_udta_offset(self) -> int:
        """Get the UDTA MP4 offset"""
        udta_offset = 0
        for i in range(1024 * 1024 * 24):
            if self.__header[i] == 117 and self.__header[i + 1] == 100 and \
                self.__header[i + 2] == 116 and self.__header[i + 3] == 97:
                udta_offset = i
                break
        return udta_offset

    def __get_gpmf_offset(self, udta_offset : int) -> [int, int]:
        """Get the GPMF MP4 offset and size"""
        offset = 0
        size = 0
        for i in range(udta_offset, udta_offset + 4096):
            if self.__header[i] == 0x47 and self.__header[i + 1] == 0x50 and \
                self.__header[i + 2] == 0x4D and self.__header[i + 3] == 0x46:
                offset = i + 4
                size = (self.__header[i - 3] << 16) + (self.__header[i - 2] << 8) + \
                    (self.__header[i - 1] << 0) - 8
                break
        return [offset, size]

    def __parse_header(self) -> None:
        """Parse the header"""
        with open(self.__filename, "rb") as file_obj:
            data = file_obj.read()
            offset = self.__get_video_header_offset(data)
            if offset > 0:
                # Valid GoPro video file
                self.__is_photo = False
                self.__header = data[offset : offset + (1024 * 1024 * 24)]
            else:
                offset, size = self.__get_photo_header_offset(data)
                if offset > 0:
                    # Valid GoPro photo file
                    self.__is_photo = True
                    self.__header = data[offset : offset + (size + 1024)]
        if not self.__header:
            # No header found—the file is invalid
            raise MetadataFileException("Unsupported file format: no header found.")
        if self.__is_photo:
            self.__gpmf_size = size
        else:
            try:
                udta_offset = self.__get_udta_offset()
            except IndexError:
                udta_offset = None
            if not udta_offset:
                raise MetadataFormatException("Unsupported file format: no UDTA offset found.")
            self.__gpmf_offset, self.__gpmf_size = self.__get_gpmf_offset(udta_offset)
        if self.__gpmf_offset == 0 and self.__gpmf_size == 0:
            raise MetadataFormatException("Unsupported file format: no GPMF offset found.")

    def __parse_type_0x63(self, data : bytes, i : int, **kwargs) -> str:  # pylint: disable=unused-private-member
        """Parse data structure 0x63 - single byte 'c' style ASCII character string"""
        value = ""
        size = kwargs['size']
        if data[i + 8] > 0:
            for j in range(0, size):
                if data[i + 8 + j] != 0 and data[i + 8 + j] != 10 and \
                    data[i + 8 + j] != 13:
                    value += bytes([data[i + 8 + j]]).decode("utf8")
                if data[i + 8 + j] == 10 or data[i + 8 + j] == 13:
                    value += "\n"
        return value

    def __parse_type_0x4c(self, data : bytes, i : int, **kwargs) -> list:
        """Parse data structure 0x4c (L) - 32-bit unsigned integer"""
        values = []
        repeat = kwargs['repeat']
        type_size = kwargs['type_size']
        if type_size > 4:
            repeat *= type_size / 4
        for k in range(0, repeat):
            num = data[i + 8 + k * 4] * 16777216
            num += (data[i + 8 + k * 4 + 1] << 16) + (data[i + 8 + k * 4 + 2] << 8) + \
                (data[i + 8 + k * 4 + 3]<<0)
            if num > 2147483647:
                # Signed numbers
                num -= 4294967296
            values.append(num)
        return values

    def __parse_type_0x6c(self, data : bytes, i : int, **kwargs) -> list:  # pylint: disable=unused-private-member
        """Parse data structure 0x6c (l) - 32-bit signed integer"""
        return self.__parse_type_0x4c(data, i, **kwargs)

    def __parse_type_0x53(self, data : bytes, i : int, **kwargs) -> list:
        """Parse data structure 0x53 (S) - 16-bit unsigned integer"""
        values = []
        repeat = kwargs['repeat']
        type_size = kwargs['type_size']
        if type_size > 2:
            repeat *= type_size / 2
        for k in range(0, repeat):
            num = (data[i + 8 + k * 2 + 0] << 8) + (data[i + 8 + k * 2 + 1] << 0)
            # Signed numbers
            if num > 32767:
                num -= 65536
            values.append(num)
        return values

    def __parse_type_0x73(self, data : bytes, i : int, **kwargs) -> list:  # pylint: disable=unused-private-member
        """Parse data structure 0x73 (s) - 16-bit signed integer"""
        return self.__parse_type_0x53(data, i, **kwargs)

    def __parse_type_0x42(self, data : bytes, i : int, **kwargs) -> list:  # pylint: disable=unused-private-member
        """Parse data structure 0x42 (B) - single byte unsigned integer"""
        values = []
        repeat = kwargs['repeat']
        type_size = kwargs['type_size']
        if type_size > 1:
            repeat *= type_size
        for k in range(0, repeat):
            num = data[i + 8 + k]
            if num > 127:
                # Signed numbers
                num -= 256
            values.append(num)
        return values

    def __parse_type_0x62(self, data : bytes, i : int, **kwargs) -> list:  # pylint: disable=unused-private-member
        """Parse data structure 0x62 (b) - single byte signed integer"""
        return self.__parse_type_0x42(data, i, **kwargs)

    def __parse_type_0x66(self, data : bytes, i : int, **kwargs) -> list:  # pylint: disable=unused-private-member
        """Parse data structure 0x66 (f) - 32-bit float (IEEE 754)"""
        values = []
        repeat = kwargs['repeat']
        type_size = kwargs['type_size']
        if type_size > 4:
            repeat *= type_size / 4
        for k in range(0, int(repeat)):
            num = data[i + 8 + k * 4] * 16777216
            num += (data[i + 8 + k * 4 + 1] << 16) + \
                (data[i + 8 + k * 4 + 2] << 8) + (data[i + 8 + k * 4 + 3] << 0)
            _float = self.__bytes_to_float(num)
            values.append(_float)
        return values

    def __parse_type_0x64(self, data : bytes, i : int, **kwargs) -> list:  # pylint: disable=unused-private-member
        """Parse data structure 0x64 (d) - 64-bit double precision (IEEE 754)"""
        values = []
        repeat = kwargs['repeat']
        type_size = kwargs['type_size']
        if type_size > 8:
            repeat *= type_size / 8
        for k in range(0, repeat):
            new_float_bytes = bytearray()
            signbit = (data[i + 8 + k * 8] & 0x80) >> 7
            # Convert 64-bit double to 32-bit float
            # Convert an 11-bit exponent to 8-bit
            expo = ((data[i + 8 + k * 8] & 0x7f) << 4) + \
                ((data[i + 8 + k * 8 + 1] & 0xf0) >> 4) - 1023
            new_expo = expo + 127
            # Extract the 23-bit mantissa from the MSBs of the double's mantissa
            new_mant23 = ((data[i + 8 + k * 8 + 1] & 0x0f) << 19) + \
                (data[i + 8 + k * 8 + 2] << 11) + (data[i + 8 + k * 8 + 3] << 3) + \
                    ((data[i + 8 + k * 8 + 4]) >> 5)
            # Reconstruct a 32-bit float
            new_float_bytes[0] = (signbit << 7) + (new_expo >> 1)
            new_float_bytes[1] = ((new_expo << 7) & 0x80) + ((new_mant23 >> 16) & 0x7f)
            new_float_bytes[2] = (new_mant23 >> 8) & 0xff
            new_float_bytes[3] = new_mant23 & 0xff
            num = new_float_bytes[0] * 16777216
            num += (new_float_bytes[1] << 16) + (new_float_bytes[2] << 8) + \
                (new_float_bytes[3] << 0)
            _float = self.__bytes_to_float(num)
            values.append(_float)
        return values

    def __parse_type_0x46(self, data : bytes, i : int, **kwargs) -> list:  # pylint: disable=unused-private-member
        """Parse data structure 0x46 (F) - 32-bit four character key (FourCC)"""
        values = []
        repeat = kwargs['repeat']
        type_size = kwargs['type_size']
        if type_size > 4:
            repeat *= type_size / 4
        for k in range(0, repeat):
            value = bytes([
                data[i + 8 + k * 4],
                data[i + 8 + k * 4 + 1],
                data[i + 8 + k * 4 + 2],
                data[i + 8 + k * 4 + 3]
            ]).decode("utf8")
            values.append(value)
        return values

    def __parse_type_0x4a(self, data : bytes, i : int, **kwargs) -> str:  # pylint: disable=unused-private-member
        """Parse data structure 0x4A (J) - 64-bit unsigned number"""
        value = "0x"
        repeat = kwargs['repeat']
        type_size = kwargs['type_size']
        if type_size > 8:
            repeat *= type_size / 8
        for k in range(0, repeat * 8):
            value += ("0" + bytes([data[i + 8 + k]]).hex()).upper()[-2:]
        return value

    def __parse_type_0x3f(self, *args, **kwargs) -> list:  # pylint: disable=unused-private-member,unused-argument
        """Parse data structure 0x3f (?) - complex"""
        return ".complex."

    def __bytes_to_float(self, bytes_to_convert) -> float:
        """Convert bytes to float"""
        sign = -1 if bytes_to_convert & 0x80000000 else 1
        exponent = ((bytes_to_convert >> 23) & 0xFF) - 127
        significand = bytes_to_convert & ~(-1 << 23)
        if exponent == 128:
            return sign * (float('nan') if significand else float('inf'))
        if exponent == -127:
            if significand == 0:
                return sign * 0.0
            exponent = -126
            significand /= (1 << 22)
        else:
            significand = (significand | (1 << 23)) / (1 << 23)
        return sign * significand * 2 ** exponent

    def __parse_metadata(self) -> None:
        """Parse the metadata in the header"""
        self.__parse_header()
        data = self.__header
        i = self.__gpmf_offset
        while i < self.__gpmf_offset + self.__gpmf_size:
            fourcc = data[i] * 16777216
            fourcc += (data[i + 1] << 16) + (data[i + 2] << 8) + (data[i + 3] << 0)
            if fourcc == 0:
                break
            type_code = data[i + 4]
            type_size = data[i + 5]
            repeat = (data[i + 6] << 8) + data[i + 7]
            size = type_size * repeat
            align_size = int((size + 3) / 4) * 4
            try:
                # FourCC
                # https://en.wikipedia.org/wiki/FourCC
                name = bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]).decode("ascii")
                # Omit names with invisible characters
                # https://www.asciitable.com/
                if not all(32 < ord(c) < 126 for c in name):
                    name = None
            except UnicodeDecodeError:
                # Invalid encoding
                name = None
            value = []
            if type_code == 0:
                i += 8
            else:
                class_name = self.__class__.__name__
                parser = getattr(self, f"_{class_name}__parse_type_{hex(type_code).lower()}", None)
                if callable(parser):
                    value = parser(
                        data,
                        i,
                        type_code = type_code,
                        size = size,
                        type_size = type_size,
                        repeat = repeat
                    )
                else:
                    value = "Unsupported data type"
                i += 8 + align_size
            if not name:
                continue
            if isinstance(value, list) and len(value) == 0:
                # Replace an empty list with an empty string
                self.__metadata.append([name, ""])
            elif isinstance(value, list) and len(value) == 1:
                # Unpack the list if there's only one value
                self.__metadata.append([name, value[0]])
            else:
                self.__metadata.append([name, value])

    def __iter__(self):
        """Return an iterator for the metadata"""
        self.__iter_counter = 0
        return self

    def __next__(self):
        """Return the next metadata item in the iterator"""
        try:
            item = self.__metadata[self.__iter_counter]
        except IndexError as exception:
            raise StopIteration from exception
        self.__iter_counter += 1
        return item


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("No filename provided")
    for _four_cc, _value in Metadata(sys.argv[1]):
        print(f"{_four_cc}: {_value}")
