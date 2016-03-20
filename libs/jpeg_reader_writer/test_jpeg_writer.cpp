#include "JPEGWriter.h"
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <vector>

struct RandomRowIter {
    RandomRowIter(const unsigned size): buffer(size) {
        operator++();
    }
    
    unsigned char* operator*() {
        return &buffer[0];
    }
    
    void operator++() {
        for (unsigned i = 0; i < buffer.size(); ++i)
            buffer[i] = rand() / (RAND_MAX - 1.0) * 255;
    }
    
    std::vector<unsigned char> buffer;
};

int main(int argc, char* argv[]) {    
    const unsigned width = 100, height = 100;
    
    std::vector<unsigned char> buffer(width * 3 * height);
    for (unsigned i = 0; i < 3 * width * height; ++i)
        buffer[i] = rand() / (RAND_MAX - 1.0) * 255;
    
    std::vector<unsigned char*> rows(height, NULL);
    for (unsigned j = 0; j < height; ++j)
        rows[j] = &buffer[j * 3 * width];
    
    unsigned quality[4] = { 10, 40, 60, 90 };
    for (unsigned i = 0; i < 4; ++i) {
        std::ostringstream filename;
        filename << "test_writer_quality_" << quality[i] << ".jpg";
        
        JPEGWriter writer;
        writer.header(width, height, 3, JPEG::COLOR_RGB);
        writer.setQuality(quality[i]);
        writer.write(filename.str(), rows.begin());
        
        if (!writer.warnings().empty())
            std::cout << "libjpeg warnings for quality " << quality[i] << ":\n" 
                      << writer.warnings() << std::endl;
    }
    
    RandomRowIter rowIter(3 * width);
    JPEGWriter writer;
    writer.header(width, height, 3, JPEG::COLOR_RGB);
    writer.write("test_writer_random_row_iter.jpg", rowIter);

}