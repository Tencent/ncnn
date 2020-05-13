#pragma once
#include <new>
#include <stddef.h>
#include <string.h>

template <typename T>
struct SimpleVector
{
    SimpleVector() {}
    SimpleVector(const size_t new_size, const T& value = T()) { resize(new_size, value); }
    ~SimpleVector() { clear(); }
    SimpleVector(const SimpleVector& v)
    {
        resize(v.size());
        for (int i = 0; i < size_; i++) { data_[i] = v.data_[i]; }
    }

    SimpleVector& operator=(const SimpleVector& v)
    {
        if (this == &v)
        {
            return *this;
        }
        resize(0);
        resize(v.size());
        for (int i = 0; i < size_; i++)
        {
            data_[i] = v.data_[i];
        }
        return *this;
    }

    void resize(const size_t new_size, const T& value = T())
    {
        try_alloc(new_size);
        if (new_size > size_)
        {
            for (int i = size_; i < new_size; i++)
            {
                new (&data_[i]) T();
            }
        }
        else if (new_size < size_)
        {
            for (int i = new_size - 1; i >= size_; i--)
            {
                data_[i].~T();
            }
        }
        size_ = new_size;
    }

    void clear()
    {
        for (int i = 0; i < size_; i++)
        {
            data_[i].~T();
        }

        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    T* data() const { return data_; }
    const size_t size() const { return size_; }
    T& operator[](int i) const { return data_[i]; }
    T* begin() const { return &data_[0]; }
    T* end() const { return &data_[size_]; }
    bool empty() const { return size_ == 0; }

    void push_back(const T& t)
    {
        try_alloc(size_ + 1);
        new (&data_[size_]) T(t);
        size_++;
    }

    void insert(T* pos, T* b, T* e)
    {
        SimpleVector* v = nullptr;
        if (b >= begin() && b < end())
        {
            //the same vector
            v = new SimpleVector(*this);
            b = v->begin() + (b - begin());
            e = v->begin() + (e - begin());
        }
        int diff = pos - begin();
        try_alloc(size_ + (e - b));
        pos = begin() + diff;
        memmove(pos + (e - b), pos, (end() - pos) * sizeof(T));
        int len = e - b;
        size_ += len;
        for (int i = 0; i < len; i++)
        {
            *pos = *b;
            pos++;
            b++;
        }
        delete v;
    }

    T* erase(T* pos)
    {
        pos->~T();
        memmove(pos, pos + 1, (end() - pos - 1) * sizeof(T));
        size_--;
        return pos;
    }

protected:
    T* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
    void try_alloc(size_t new_size)
    {
        if (new_size * 3 / 2 > capacity_ / 2)
        {
            capacity_ = new_size * 2;
            T* new_data = (T*)new char[capacity_ * sizeof(T)];
            memset(new_data, 0, capacity_ * sizeof(T));
            if (data_)
            {
                memmove(new_data, data_, sizeof(T) * size_);
                delete[](char*) data_;
            }
            data_ = new_data;
        }
    }
};

struct SimpleString : public SimpleVector<char>
{
    SimpleString() {}
    SimpleString(const char* str)
    {
        int len = strlen(str);
        resize(len);
        memcpy(data_, str, len);
    }
    const char* c_str() const { return (const char*)data_; }
    bool operator==(const SimpleString& str2) const { return strcmp(data_, str2.data_) == 0; }
    bool operator==(const char* str2) const { return strcmp(data_, str2) == 0; }
    bool operator!=(const char* str2) const { return strcmp(data_, str2) != 0; }
    SimpleString& operator+=(const SimpleString& str1)
    {
        insert(end(), str1.begin(), str1.end());
        return *this;
    }
};

inline SimpleString operator+(const SimpleString& str1, const SimpleString& str2)
{
    SimpleString str(str1);
    str.insert(str.end(), str2.begin(), str2.end());
    return str;
}

