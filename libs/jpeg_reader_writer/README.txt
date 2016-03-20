JPEGReader/JPEGWriter are C++ wrappers around the widely-supported libjpeg 
library by the Independent JPEG Group (http://www.ijg.org/).

It's only dependency is the C++ Standard Template Library (STL) and libjpeg
iteself.  The code uses a little bit of the STL (std::vector, std::string) and
throws exceptions on fatal errors from libjpeg.  It is thread-safe if two
threads never access the same JPEGLoader object at the same time.  It is
intended to be used by compiling the code directly into your application.

It exposes the following functionality from libjpeg for reading images:
- Querying the header of an image before committing to a full load
- Querying size, number of color components and colorspace
- Loading directly into a user-supplied buffer
- Loading an entire image at once or a fixed number of rows at a time
- Changing the destination color space before loading (useful for fast
  greyscale previews)
- Extracting only 1/2-, 1/4-, or 1/8-scale images (useful for fast previews)
- Trading off algorithmic decompression quality versus speed
- Quantizing to a fixed number of colors

It exposes the following functionality from libjpeg for writing images:
- Setting the size, number of color components and color space of image
- Setting the compression quality
- Trading off algorithmic compression quality versus speed
- Writing an entire image at once or a fixed number of rows at a time

In addition:
- Errors are reported by throwing exceptions
- libjpeg output is collected internally instead of being written to
  std::cout, and can be queried at any time.

test_jpeg_reader.cpp is included to show how to use the library and to time
image loading with various options.  test_jpeg_writer is included to show how
to write JPEG images.  A trivial IDE project is provided for
XCode 2.4.  Code documentation is provided in html/annotated.html.

The author of this work is Adrian Secord.
Contact: http://www.cs.nyu.edu/~ajsecord/

This work is hereby released into the Public Domain. To view a copy of the 
public domain dedication, visit 
http://creativecommons.org/licenses/publicdomain/ 
or send a letter to Creative Commons, 543 Howard Street, 5th Floor, 
San Francisco, California, 94105, USA.

Attributing the author is appreciated but not required. 
