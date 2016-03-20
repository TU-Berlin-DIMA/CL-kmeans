#include "JPEGReader.h"
#include <stdexcept>
#include <cassert>
#include <limits>

#include <iostream> // debug

// We need version 6b
#if JPEG_LIB_VERSION < 62
#error JPEGReader needs IJG libjpeg with a version of at least 6b.
#endif

namespace {
    // Bounce errors to the object
    void libjpeg_error_exit(j_common_ptr cinfo) {
        assert(cinfo->is_decompressor && cinfo->client_data);
        ((JPEGReader*) cinfo->client_data)->error_exit();
    }
    
    void libjpeg_output_message(j_common_ptr cinfo) {
        assert(cinfo->is_decompressor && cinfo->client_data);
        ((JPEGReader*) cinfo->client_data)->output_message();
    }
}

JPEGReader::JPEGReader():
    file(NULL),
    max_row_ptrs(std::numeric_limits<unsigned>::max()) {
    
    // Error handling first, in case the initialization fails.
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = libjpeg_error_exit;
    jerr.output_message = libjpeg_output_message;
    
    // Initialize the decompression part.
    jpeg_create_decompress(&cinfo);
    assert(sizeof(JSAMPLE) == 1);

    cinfo.client_data = this;
}

JPEGReader::~JPEGReader() {
    assert(cinfo.client_data == this);
    
    jpeg_destroy_decompress(&cinfo);
    if (file) 
        fclose(file);
}

void JPEGReader::header(const std::string& path) {
    warningMsg.clear();
    
    // Specify the source of the compressed data (eg, a file)
    if ((file = fopen(path.c_str(), "rb")) == NULL)
        throw std::runtime_error("Cannot open " + path);
    
    jpeg_stdio_src(&cinfo, file);
    
    // Call jpeg_read_header to obtain image info
    jpeg_read_header(&cinfo, true);
    
    // Make sure that output_width, output_height, out_components get set.
    jpeg_calc_output_dimensions(&cinfo);
    assert(cinfo.output_components == cinfo.out_color_components);  // Ensure we're not color-mapped
}

JPEG::Scale JPEGReader::scale() const {
    assert(cinfo.scale_num == 1);
    switch (cinfo.scale_denom) {
        case 1: 
        case 2: 
        case 4: 
        case 8: 
            return (JPEG::Scale) cinfo.scale_denom;
            break;
            
        default:
            assert(!"Unknown libjpeg output scale");
            return JPEG::SCALE_FULL_SIZE;
    }
}

void JPEGReader::setScale(const JPEG::Scale value) {
    cinfo.scale_num = 1;
    cinfo.scale_denom = value;
    
    jpeg_calc_output_dimensions(&cinfo);
}

void JPEGReader::chooseGoodScale(const unsigned targetWidth, const unsigned targetHeight) {
    cinfo.scale_num = 1;
    
    unsigned denom = 16;
    do {
        denom /= 2;
        cinfo.scale_denom = denom;
        jpeg_calc_output_dimensions(&cinfo);
    } while (denom > 1 && (width() < targetWidth || height() < targetHeight));

    assert((width() >= targetWidth && height() >= targetHeight) || cinfo.scale_denom == 1);
}

JPEG::ColorSpace JPEGReader::colorSpace() const {
    return (JPEG::ColorSpace) cinfo.out_color_space;
}

void JPEGReader::setColorSpace(const JPEG::ColorSpace value) {
    cinfo.out_color_space = (J_COLOR_SPACE) value;
    jpeg_calc_output_dimensions(&cinfo);
}

JPEG::Dither JPEGReader::dither() const {
    return (JPEG::Dither) cinfo.dither_mode;
}

void JPEGReader::setDither(const JPEG::Dither& value) {
    cinfo.dither_mode = (J_DITHER_MODE) value;
    jpeg_calc_output_dimensions(&cinfo);
}

unsigned JPEGReader::quantization() const {
    return (cinfo.quantize_colors ? cinfo.desired_number_of_colors : 0);
}

void JPEGReader::setQuantization(const unsigned value) {
    if (value > 0) {
        cinfo.quantize_colors = true;
        cinfo.desired_number_of_colors = value;
        cinfo.two_pass_quantize = true;
        
    } else {
        cinfo.quantize_colors = false;
    }

    jpeg_calc_output_dimensions(&cinfo);
}

void JPEGReader::setTradeoff(const JPEG::TimeQualityTradeoff value) {
    switch (value) {
        case JPEG::FASTER:
            cinfo.dct_method = JDCT_FASTEST;
            cinfo.do_fancy_upsampling = false;
            break;
            
        case JPEG::DEFAULT:
            cinfo.dct_method = JDCT_DEFAULT;
            cinfo.do_fancy_upsampling = true;
            break;
            
        case JPEG::BETTER:
            cinfo.dct_method = JDCT_FLOAT;
            cinfo.do_fancy_upsampling = true;
            break;
    
        default:
            assert(!"Invalid time/quality tradeoff.");
            break;
    }
}

void JPEGReader::error_exit() {
    output_message();
    throw std::runtime_error("libjpeg error: " + warningMsg);
}

void JPEGReader::output_message() {

    // Use the default routine to generate the message.
    char buffer[JMSG_LENGTH_MAX];
    (cinfo.err->format_message) ((jpeg_common_struct*) &cinfo, buffer);
    
    warningMsg += buffer;
    warningMsg += '\n';
}


