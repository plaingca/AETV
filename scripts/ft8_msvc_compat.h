#ifndef AETV_FT8_MSVC_COMPAT_H
#define AETV_FT8_MSVC_COMPAT_H
#include <string.h>
static __inline char* stpcpy(char* destination, const char* source)
{
    size_t length = strlen(source);
    memcpy(destination, source, length + 1);
    return destination + length;
}
#endif
