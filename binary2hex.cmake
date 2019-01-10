
# convert to binary to hex data, unsigned int, little endian
file(READ ${BINARY_FILE} bytes HEX)
string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1, " uints "${bytes}")
file(WRITE "${BINARY_FILE}.hex.h" "${uints}")
