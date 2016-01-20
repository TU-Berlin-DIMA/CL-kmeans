#ifndef CSV_HPP
#define CSV_HPP

#include <vector>
#include <tuple>
#include <iostream>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <cassert>
#include <array>
#include <type_traits>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

// #define PARSE_WITH_BOOST

#ifdef PARSE_WITH_BOOST
#include <boost/lexical_cast.hpp>
#endif

#include <boost/spirit/include/qi.hpp>

namespace cle {

class CSV {
public:
    template <typename T, typename ... Ts>
    int read_csv(char const *file_name, std::vector<T>& vector, Ts& ... vectors) {

        constexpr size_t NUM_VECTORS = sizeof...(Ts) + 1;
        CToken tokens[NUM_VECTORS];
        size_t file_size = 0;
        char const * mapped = NULL;
        size_t file_offset = 0;
        size_t chars_tokenized = 0;
        std::stringstream ssbuf;

        // Open and mmap file
        open_file(file_name, mapped, file_size);

        // Preallocate memory
        tune_vectors((file_size + NUM_VECTORS) / NUM_VECTORS,
                vector, vectors ...);

        // Process file
        file_offset = 0;
        while (file_offset < file_size) {
            chars_tokenized = tokenize(&mapped[file_offset], file_size, tokens,
                    NUM_VECTORS, delimiter_);

            if (chars_tokenized != 0) {
                parse_line(ssbuf, tokens, vector, vectors...);
            }

            file_offset += chars_tokenized;
        }

        close_file(mapped, file_size);

        return 1;
    }

    template <typename T, size_t size>
    int read_csv(char const *file_name, std::array<std::vector<T>, size>& vectors) {

        size_t file_size = 0;
        char const * mapped = NULL;
        size_t file_offset = 0;
        size_t chars_tokenized = 0;
        CToken tokens[size];
        std::stringstream ssbuf;

        open_file(file_name, mapped, file_size);

        // Process file
        file_offset = 0;
        while (file_offset < file_size) {
            chars_tokenized = tokenize(&mapped[file_offset], file_size, tokens,
                    size, delimiter_);

            if (chars_tokenized != 0) {
                parse_line(ssbuf, tokens, vectors);
            }

            file_offset += chars_tokenized;
        }

        close_file(mapped, file_size);

        return 1;
    }

    void set_delimiter(const char delimiter) {
        delimiter_ = delimiter;
    }

private:
    using Token = std::tuple<char*,size_t>;
    using CToken = std::tuple<char const*,size_t>;

    int open_file(char const* file_name, char const*& mapped, size_t& file_size) {
        struct stat file_stat = {};
        int fd = 0;
        int status = 0;

        fd = open(file_name, O_RDONLY);
        if (fd < 0) {
            std::cerr << "Open file " << file_name << " failed with code "
                << strerror(errno) << std::endl;
            return fd;
        }

        status = fstat(fd, &file_stat);
        if (status < 0) {
            std::cerr << "Stat file " << file_name << " failed with code "
                << strerror(errno) << std::endl;
            return status;
        }
        file_size = file_stat.st_size;

        mapped = (char*) mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) {
            std::cerr << "Mmap file " << file_name << " failed with code "
                << strerror(errno) << std::endl;
            return errno;
        }

        if(posix_madvise((void*)mapped, file_size, POSIX_MADV_WILLNEED) != 0) {
            std::cerr << "Madvise failed with code "
                << strerror(errno) << std::endl;
            return errno;
        }

        return 1;
    }

    int close_file(char const * const mapped, size_t const file_size) {
        int ret = 0;

        ret = munmap((void*)mapped, file_size);
        if (ret < 0) {
            std::cerr << "Munmap failed with code "
                << strerror(errno) << std::endl;
        }

        return ret;
    }

    size_t tokenize(char const *buffer, size_t buffer_size,
            CToken *tokens, size_t num_tokens, char delimiter) {

        size_t offset = 0;
        size_t begin_offset = 0;
        size_t t = 0;
        while (offset < buffer_size && t < num_tokens) {
            char const& c = buffer[offset];

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
        char const * begin = std::get<0>(tokens[depth]);
        size_t size = std::get<1>(tokens[depth]);
        char const * last = &begin[size];

        if (std::is_same<double, T>::value) {
            boost::spirit::qi::parse(begin, last, boost::spirit::qi::double_, v);
            vec.push_back(v);
        }
        else {
#ifdef PARSE_WITH_BOOST
            v = boost::lexical_cast<T>(begin, size);
            vec.push_back(v);
#else
            ssbuf.write(begin, size);
            ssbuf >> v;
            vec.push_back(v);
            ssbuf.clear();
            ssbuf.str("");
#endif
        }

        return 1;
    }

    template <size_t depth = 0, typename T, typename ... Ts>
    int parse_line(std::stringstream& ssbuf, CToken const *tokens,
            std::vector<T>& vec, Ts& ... other) {

        T v;
        char const * begin = std::get<0>(tokens[depth]);
        size_t size = std::get<1>(tokens[depth]);
        char const * last = &begin[size];

        if (std::is_same<double, T>::value) {
            boost::spirit::qi::parse(begin, last, boost::spirit::qi::double_, v);
            vec.push_back(v);
        }
        else {
#ifdef PARSE_WITH_BOOST
            v = boost::lexical_cast<T>(begin, size);
            vec.push_back(v);
#else
            ssbuf.write(begin, size);
            ssbuf >> v;
            vec.push_back(v);
            ssbuf.clear();
            ssbuf.str("");
#endif
        }

        parse_line<depth+1>(ssbuf, tokens, other...);

        return 1;
    }

    template <typename T, size_t size>
    int parse_line(std::stringstream& ssbuf, CToken const *tokens,
            std::array<std::vector<T>, size>& vectors) {

        T v;
        char const * token_begin;
        char const * token_last;
        size_t token_size;

        for (size_t i = 0; i != size; ++i) {
            token_begin = std::get<0>(tokens[i]);
            token_size = std::get<1>(tokens[i]);
            token_last = &token_begin[token_size];

            if (std::is_same<double, T>::value) {
                boost::spirit::qi::parse(token_begin, token_last, boost::spirit::qi::double_, v);
                vectors[i].push_back(v);
            }
            else {
#ifdef PARSE_WITH_BOOST
                v = boost::lexical_cast<T>(token_begin, token_size);
                vectors[i].push_back(v);
#else
                ssbuf.write(token_begin, token_size);
                ssbuf >> v;
                vectors[i].push_back(v);
                ssbuf.clear();
                ssbuf.str("");
#endif
            }
        }

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

    template <typename T, size_t size>
    void tune_vectors(size_t avg_size, std::array<std::vector<T>, size>& vectors) {
        for (std::vector<T>& v : vectors) {
            tune_vectors(avg_size, v);
        }

    }

    char delimiter_ = ',';
};

}

#endif /* CSV_HPP */
