if __name__ == '__main__':
    print("What Unicode character does chr(0) return?")
    print("text" + chr(0) + "text")
    print("A: null byte")
    print("How does this character’s string representation (__repr__()) differ from its printed representation?")
    print(chr(0).__str__())
    print(chr(0).__repr__())
    print("A: repr is the byte level representation")
    print("this is a test" + chr(0) + "string")
    print("-----")
    test_string = "hello! こんにちは!"
    print("UTF-8")
    utf8_encoded = test_string.encode("utf-8")
    print(utf8_encoded)
    print(list(utf8_encoded))
    print(type(utf8_encoded))
    print(len(test_string))
    print(len(utf8_encoded))

    print("UTF-16")
    utf8_encoded = test_string.encode("utf-32")
    print(utf8_encoded)
    print(list(utf8_encoded))
    print(type(utf8_encoded))
    print(len(test_string))
    print(len(utf8_encoded))

    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    
    print(0xc279)
    print((0xc279).to_bytes(2, 'big').decode("utf-8"))
