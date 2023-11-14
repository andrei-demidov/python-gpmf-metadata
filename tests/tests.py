"""Unit tests"""

import unittest

from gpmf_metadata import Metadata
from gpmf_metadata.exceptions import MetadataFileException

from .fixtures.gopro_photo import METADATA as GO_PRO_PHOTO_METADATA
from .fixtures.gopro_video import METADATA as GO_PRO_VIDEO_METADATA


FIXTURES = {
    'gopro_photo': "tests/fixtures/gopro_photo.jpg",
    'gopro_video': "tests/fixtures/gopro_video.mp4",
    'not_gopro_photo': "tests/fixtures/not_gopro_photo.jpg",
    'not_gopro_video': "tests/fixtures/not_gopro_video.mp4",
    'invalid_video': "tests/fixtures/invalid_video.mp4",
    'nonexistent_file': "tests/fixtures/404"
}


class TestNonexistentFile(unittest.TestCase):
    """Test case when the file doesn't exist"""
    def test_nonexistent_file(self):
        """Test case when the file doesn't exist"""
        with self.assertRaises(MetadataFileException):
            Metadata(FIXTURES['nonexistent_file'])


class TestInvalidVideo(unittest.TestCase):
    """Test case when the video is invalid"""
    def test_invalid_video(self):
        """Test case when the video is invalid"""
        with self.assertRaises(MetadataFileException):
            Metadata(FIXTURES['invalid_video'])


class TestNotGoProPhoto(unittest.TestCase):
    """Test case when the photo is from another camera"""
    def test_not_go_pro_photo(self):
        """Test case when the photo is from another camera"""
        with self.assertRaises(MetadataFileException):
            Metadata(FIXTURES['not_gopro_photo'])


class TestNotGoProVideo(unittest.TestCase):
    """Test case when the video is from another camera"""
    def test_not_go_pro_video(self):
        """Test case when the video is from another camera"""
        with self.assertRaises(MetadataFileException):
            Metadata(FIXTURES['not_gopro_video'])


class TestGoProPhoto(unittest.TestCase):
    """Test case when the photo is supported and valid"""
    def test_go_pro_photo(self):
        """Test case when the photo is supported and valid"""
        self.assertEqual(
            Metadata(FIXTURES['gopro_photo']).metadata,
            GO_PRO_PHOTO_METADATA
        )


class TestGoProVideo(unittest.TestCase):
    """Test case when the video is supported and valid"""
    def test_go_pro_video(self):
        """Test case when the video is supported and valid"""
        self.assertEqual(
            Metadata(FIXTURES['gopro_video']).metadata,
            GO_PRO_VIDEO_METADATA
        )


if __name__ == '__main__':
    unittest.main()
