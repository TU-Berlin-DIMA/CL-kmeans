/// Test and time libjpeg and JPEGReader.
///
/// Timing machine specifications: 
///
/// Processor Name:             Intel Core 2 Duo
/// Processor Speed:            2.16 GHz
/// L2 Cache (per processor):   4 MB
/// Memory:                     2 GB
/// Bus Speed:                  667 MHz
///
/// Input image was a 2048x1536 camera JPEG with a disk size of around 900K.
/// Only a single core was used in this test.
///
/// Compiler: i686-apple-darwin8-gcc-4.0.1 (GCC) 4.0.1 (Apple Computer, Inc. build 5363)
/// Flags: -g -O3
///
/// Timings: 100 reads from disk, no writing to the disk at all (image is just discarded).
/// Times reported are user + system time.
/// From slowest to fastest, including command-line and time *per image*:
///
/// -b                  0.169145s       (use big image-sized buffer)
///                     0.144526s       (full-sized, default quality)
/// -f                  0.120569s       (full-sized, lower quality)
/// -g                  0.0764692s      (only extract greyscale image)
/// -r                  0.0825156s      (extract 1/2-sized image)
/// -r -r               0.0531246s      (extract 1/4-sized image)
/// -r -r -r            0.0294817s      (extract 1/8-sized image)
/// -b -g -f -r -r -r   0.028011s       (big buffer, greyscale, lower quality, 1/8-sized image)
/// -g -f -r -r -r      0.0278272s      (greyscale, lower quality, 1/8-sized image)
///
/// The biggest win is to just extract a smaller image.
/// Next biggest is to extract only the greyscale information.
/// Allocating a huge buffer at once and filling it generally isn't worth it 
/// compared to processing things a row at a time or so.

#include "JPEGReader.h"

#include <iostream>
#include <fstream>
#include <cassert>

#include <sys/time.h>
#include <sys/resource.h>

double cpuUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    const double user = usage.ru_utime.tv_sec + 1e-6 * usage.ru_utime.tv_usec;
    const double sys  = usage.ru_stime.tv_sec + 1e-6 * usage.ru_stime.tv_usec;
    
    return user + sys;
}

struct OutputRowIter {
    OutputRowIter(std::ostream& out, const unsigned size): o(out), buffer(size) {}
    
    unsigned char* operator*() {
        return &buffer[0];
    }
    
    OutputRowIter& operator++() {
        o.write((char*)(&buffer[0]), buffer.size());
        return *this;
    }
    
    std::ostream& o;
    std::vector<unsigned char> buffer;
};

struct NullRowIter {
    NullRowIter(const unsigned size): buffer(size) {}
    
    unsigned char* operator*() {
        return &buffer[0];
    }
    
    NullRowIter& operator++() {
        return *this;
    }
    
    std::vector<unsigned char> buffer;
};

int main (int argc, char * const argv[]) {
    assert(argc > 1);
    
    std::string filename;
    bool timing = false;
    bool big_buffer = false;
    bool greyscale = false;
    bool fast = false;
    bool verbose = false;
    bool read_only = false;
    bool dither = false;
    unsigned reduction = 0;
    
    std::cout << "Command line: ";
    for (int i = 1; i < argc; ++i) {
        std::cout << argv[i] << ' ';
        
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
                case 't': timing = !timing; break;
                case 'b': big_buffer = !big_buffer; break;
                case 'g': greyscale = !greyscale;   break;
                case 'f': fast = !fast;             break;
                case 'r': ++reduction;              break;
                case 'v': verbose = !verbose;       break;
                case 'n': read_only = !read_only;   break;
                case 'd': dither = !dither;         break;
                default:
                    std::cerr << "I don't know what you're talking about." << std::endl;
                    return -1;
            }
        } else {
            filename = argv[i];
        }
    }
    std::cout << '\n';
    
    if (filename.empty()) {
        std::cerr << "Need at least a filename" << std::endl;
        return -1;
    }   

    reduction = std::min(reduction, 3U);
    
    if (timing)     std::cout << "Timing multiple runs" << std::endl;
    if (big_buffer) std::cout << "Using big buffer for entire image" << std::endl;
    if (greyscale)  std::cout << "Converting to greyscale" << std::endl;
    if (fast)       std::cout << "Setting faster mode" << std::endl;
    if (reduction)  std::cout << "Reducing size " << reduction << " times" << std::endl;
    if (read_only)  std::cout << "Only reading, not writing" << std::endl;
    if (dither)     std::cout << "Dithering" << std::endl;
    
    const double start = cpuUsage();
    
    JPEGReader loader;
    const unsigned num_iters = (timing ? 100 : 1);
    for (unsigned iter = 0; iter < num_iters; ++iter) {
        loader.header(filename);
        
        if (verbose)
            std::cout << "Size: " << loader.width() << 'x' << loader.height() << '\n'
                      << "Number of color channels: " << loader.components() << std::endl;

        if (greyscale) 
            loader.setColorSpace(JPEG::COLOR_GRAYSCALE);
        
        if (reduction) 
            loader.setScale((JPEG::Scale)(JPEG::SCALE_FULL_SIZE + reduction));
        
        if (fast) 
            loader.setTradeoff(JPEG::FASTER);
        
        if (dither) {
            loader.setQuantization(64);
            loader.setDither(JPEG::DITHER_NONE);
        }
    
        // Start a PPM file
        std::ofstream* ppm = NULL;
        if (!read_only) {
            ppm = new std::ofstream(loader.components() == 3 ? "output.ppm" : "output.pgm");
            *ppm << (loader.components() == 3 ? "P6" : "P5") << '\n'
                 << loader.width() << ' ' << loader.height() << '\n'
                 << 255 << '\n';
        }
    
        if (big_buffer) {
            std::vector<unsigned char> buffer(loader.width() * loader.height() * loader.components());
            std::vector<unsigned char*> rows(loader.height(), NULL);
            for (unsigned i = 0; i < loader.height(); ++i) 
                rows[i] = &buffer[i * loader.width() * loader.components()];
            
            loader.load(rows.begin());
            
            if (ppm) 
                ppm->write((char*)(&buffer[0]), buffer.size());
            
        } else if (!read_only) {
            OutputRowIter rowIter(*ppm, loader.width() * loader.components());
            loader.setMaxRowPtrs(1);
            loader.load(rowIter);
            ++rowIter;  // Dump the last row
                   
        } else {
            assert(!big_buffer && read_only);
            NullRowIter rowIter(loader.width() * loader.components());
            loader.setMaxRowPtrs(1);
            loader.load(rowIter);
        }
        
        if (!read_only) {
            assert(*ppm);
            delete ppm;
        }

        if (!loader.warnings().empty())
            std::cerr << "libjpeg warnings: " << loader.warnings() << std::endl;
    }
    
    const double stop = cpuUsage();
    std::cout << "Time per image: " << (stop - start) / num_iters << 's' << std::endl;

    return 0;
}
