#ifndef CSV_HPP
#define CSV_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <cassert>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace cle {

class CSV {
public:
    template <typename ... Ts>
    int read_csv(char const *file_name, Ts& ... vectors) {

        constexpr size_t NUM_VECTORS = sizeof...(Ts);
        CToken tokens[NUM_VECTORS];
        struct stat file_stat;
        int fd = 0;
        size_t file_size = 0;
        int status = 0;
        char const * mapped = NULL;
        size_t file_offset = 0;
        size_t chars_tokenized = 0;
        std::stringstream ssbuf;

        // Open and mmap file
        fd = open(file_name, O_RDONLY);
        if (fd < 0) {
            std::cerr << "Open file " << file_name << " failed with code "
                << strerror(errno) << std::endl;
        }

        status = fstat(fd, &file_stat);
        if (status < 0) {
            std::cerr << "Stat file " << file_name << " failed with code "
                << strerror(errno) << std::endl;
        }
        file_size = file_stat.st_size;

        mapped = (char*) mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) {
            std::cerr << "Mmap file " << file_name << " failed with code "
                << strerror(errno) << std::endl;
        }

        if(posix_madvise((void*)mapped, file_size, POSIX_MADV_WILLNEED) != 0) {
            std::cerr << "Madvise failed with code "
                << strerror(errno) << std::endl;
        }

        // Preallocate memory
        tune_vectors((file_size + NUM_VECTORS) / NUM_VECTORS,
                vectors ...);

        // Process file
        file_offset = 0;
        while (file_offset < file_size) {
            chars_tokenized = tokenize(&mapped[file_offset], file_size, tokens,
                    NUM_VECTORS, delimiter_);

            if (chars_tokenized != 0) {
                parse_line(ssbuf, tokens, vectors...);
            }

            file_offset += chars_tokenized;
        }

        munmap((void*)mapped, file_size);

        return 1;
    }

    void set_delimiter(const char delimiter) {
        delimiter_ = delimiter;
    }

private:
    using Token = std::tuple<char*,size_t>;
    using CToken = std::tuple<char const*,size_t>;

    size_t tokenize(char const *buffer, size_t buffer_size,
            CToken *tokens, size_t num_tokens, char delimiter) {

        size_t offset = 0;
        size_t begin_offset = 0;
        size_t t = 0;
        while (offset < buffer_size && t < num_tokens) {
            char c = buffer[offset];

            if (c == delimiter || c == '\n') {
                std::get<0>(tokens[t]) = &buffer[begin_offset];
                std::get<1>(tokens[t]) = offset - begin_offset;
                begin_offset = offset + 1;

                ++t;
            }

            ++offset;
        }

        if (offset == buffer_size && t < num_tokens) {
            std::get<0>(tokens[t]) = &buffer[begin_offset];
            std::get<1>(tokens[t]) = offset - begin_offset;
        }

        assert(t == num_tokens || t == 0);

        return offset;
    }

    template <size_t depth = 0, typename T>
    int parse_line(std::stringstream& ssbuf, CToken const *tokens,
            std::vector<T>& vec) {

        T v;

        ssbuf.write(std::get<0>(tokens[depth]), std::get<1>(tokens[depth]));
        ssbuf >> v;

        vec.push_back(v);
        ssbuf.clear();
        ssbuf.str("");

        return 1;
    }

    template <size_t depth = 0, typename T, typename ... Ts>
    int parse_line(std::stringstream& ssbuf, CToken const *tokens,
            std::vector<T>& vec, Ts& ... other) {

        T v;

        ssbuf.write(std::get<0>(tokens[depth]), std::get<1>(tokens[depth]));
        ssbuf >> v;
        vec.push_back(v);
        ssbuf.clear();
        ssbuf.str("");

        parse_line<depth+1>(ssbuf, tokens, other...);

        return 1;
    }

    template <typename T>
    void tune_vectors(size_t avg_size, std::vector<T>& vec) {
        vec.reserve(avg_size / sizeof(T));
    }

    template <typename T, typename ... Ts>
    void tune_vectors(size_t avg_size, std::vector<T>& vec, Ts& ... other) {
        vec.reserve(avg_size / sizeof(T));

        tune_vectors(avg_size, other ...);
    }

    char delimiter_ = ',';
};

};

#endif /* CSV_HPP */
