from collections import List
from pathlib import Path


struct JsonValueTag:
    alias Null = 0
    alias Bool = 1
    alias Number = 2
    alias String = 3
    alias Array = 4
    alias Object = 5


struct JsonTokenTag:
    alias Invalid = 0
    alias LeftBrace = 1
    alias RightBrace = 2
    alias LeftBracket = 3
    alias RightBracket = 4
    alias Colon = 5
    alias Comma = 6
    alias String = 7
    alias Number = 8
    alias TrueLit = 9
    alias FalseLit = 10
    alias NullLit = 11
    alias EOF = 12


struct JsonParseDiagnostic(Movable, ImplicitlyCopyable):
    var message: String
    var offset: Int
    var line: Int
    var column: Int
    var context: String
    var caret_column: Int

    fn __init__(out self):
        self.message = ""
        self.offset = 0
        self.line = 1
        self.column = 1
        self.context = ""
        self.caret_column = 1

    fn render(self) -> String:
        var rendered = (
            "JSON parse error at line "
            + String(self.line)
            + ", column "
            + String(self.column)
            + ": "
            + self.message
        )
        if len(self.context) == 0:
            return rendered
        rendered += "\n"
        rendered += self.context
        rendered += "\n"
        for _ in range(self.caret_column - 1):
            rendered += " "
        rendered += "^"
        return rendered


struct JsonToken(Movable, ImplicitlyCopyable):
    var tag: Int
    var start: Int
    var end: Int
    var value_start: Int
    var value_end: Int
    var has_escape: Bool
    var number_is_integer: Bool

    fn __init__(out self):
        self.tag = JsonTokenTag.Invalid
        self.start = 0
        self.end = 0
        self.value_start = 0
        self.value_end = 0
        self.has_escape = False
        self.number_is_integer = False


struct JsonObjectEntry(Movable, ImplicitlyCopyable):
    var key: String
    var value_index: Int
    var key_offset: Int

    fn __init__(out self, key: String, value_index: Int, key_offset: Int):
        self.key = key
        self.value_index = value_index
        self.key_offset = key_offset


struct JsonNode(Movable, ImplicitlyCopyable):
    var tag: Int
    var start: Int
    var end: Int
    var bool_value: Bool
    var number_value: Float64
    var number_is_integer: Bool
    var text_value: String
    var child_offset: Int
    var child_count: Int

    fn __init__(out self):
        self.tag = JsonValueTag.Null
        self.start = 0
        self.end = 0
        self.bool_value = False
        self.number_value = 0.0
        self.number_is_integer = False
        self.text_value = ""
        self.child_offset = 0
        self.child_count = 0


struct JsonDocument(Movable):
    var text: String
    var nodes: List[JsonNode]
    var object_entries: List[JsonObjectEntry]
    var array_items: List[Int]
    var root_index: Int

    fn __init__(out self):
        self.text = ""
        self.nodes = List[JsonNode]()
        self.object_entries = List[JsonObjectEntry]()
        self.array_items = List[Int]()
        self.root_index = -1

    fn __init__(out self, text: String):
        self.text = String(text)
        self.nodes = List[JsonNode]()
        self.object_entries = List[JsonObjectEntry]()
        self.array_items = List[Int]()
        self.root_index = -1

    fn add_node(mut self, node: JsonNode) -> Int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    fn root(self) raises -> JsonNode:
        if self.root_index < 0:
            raise Error("JSON document has no root node")
        return self.nodes[self.root_index]

    fn object_find(self, node_index: Int, key: StringSlice) raises -> Int:
        var node = self.nodes[node_index]
        if node.tag != JsonValueTag.Object:
            raise Error("JSON node is not an object")
        for i in range(node.child_count):
            var entry = self.object_entries[node.child_offset + i]
            if entry.key == key:
                return entry.value_index
        return -1

    fn object_entry(
        self, node_index: Int, item_index: Int
    ) raises -> JsonObjectEntry:
        var node = self.nodes[node_index]
        if node.tag != JsonValueTag.Object:
            raise Error("JSON node is not an object")
        if item_index < 0 or item_index >= node.child_count:
            raise Error("JSON object entry index out of range")
        return self.object_entries[node.child_offset + item_index]

    fn array_item(self, node_index: Int, item_index: Int) raises -> Int:
        var node = self.nodes[node_index]
        if node.tag != JsonValueTag.Array:
            raise Error("JSON node is not an array")
        if item_index < 0 or item_index >= node.child_count:
            raise Error("JSON array index out of range")
        return self.array_items[node.child_offset + item_index]

    fn raw_slice(self, start: Int, end: Int) -> String:
        return String(StringSlice(unsafe_from_utf8=self.text.as_bytes()[start:end]))

    fn node_text(self, node_index: Int) -> String:
        return self.nodes[node_index].text_value

    fn node_number(self, node_index: Int) -> Float64:
        return self.nodes[node_index].number_value

    fn node_bool(self, node_index: Int) -> Bool:
        return self.nodes[node_index].bool_value

    fn node_len(self, node_index: Int) -> Int:
        return self.nodes[node_index].child_count

    fn node_tag(self, node_index: Int) -> Int:
        return self.nodes[node_index].tag

    fn node_offset(self, node_index: Int) -> Int:
        return self.nodes[node_index].start
fn _is_whitespace(ch: Byte) -> Bool:
    return (
        ch == Byte(ord(" "))
        or ch == Byte(ord("\n"))
        or ch == Byte(ord("\r"))
        or ch == Byte(ord("\t"))
    )


fn _is_digit(ch: Byte) -> Bool:
    return ch >= Byte(ord("0")) and ch <= Byte(ord("9"))


fn _is_hex_digit(ch: Byte) -> Bool:
    return (
        _is_digit(ch)
        or (ch >= Byte(ord("a")) and ch <= Byte(ord("f")))
        or (ch >= Byte(ord("A")) and ch <= Byte(ord("F")))
    )


fn _hex_value(ch: Byte) -> Int:
    if ch >= Byte(ord("0")) and ch <= Byte(ord("9")):
        return Int(ch) - ord("0")
    if ch >= Byte(ord("a")) and ch <= Byte(ord("f")):
        return 10 + Int(ch) - ord("a")
    if ch >= Byte(ord("A")) and ch <= Byte(ord("F")):
        return 10 + Int(ch) - ord("A")
    return -1


fn _append_utf8_segment(mut out: String, text: String, start: Int, end: Int):
    if end <= start:
        return
    out += String(StringSlice(unsafe_from_utf8=text.as_bytes()[start:end]))


struct JsonParser(Movable):
    var text: String
    var position: Int
    var current: JsonToken

    fn __init__(out self, text: String):
        self.text = String(text)
        self.position = 0
        self.current = JsonToken()

    fn _make_diagnostic(self, offset: Int, message: String) -> JsonParseDiagnostic:
        var diagnostic = JsonParseDiagnostic()
        diagnostic.message = message
        diagnostic.offset = offset

        var bytes = self.text.as_bytes()
        var bounded_offset = offset
        if bounded_offset < 0:
            bounded_offset = 0
        if bounded_offset > len(bytes):
            bounded_offset = len(bytes)

        var line = 1
        var column = 1
        var line_start = 0
        for i in range(bounded_offset):
            if bytes[i] == Byte(ord("\n")):
                line += 1
                column = 1
                line_start = i + 1
            else:
                column += 1
        diagnostic.line = line
        diagnostic.column = column

        var line_end = len(bytes)
        for i in range(line_start, len(bytes)):
            if bytes[i] == Byte(ord("\n")):
                line_end = i
                break
        diagnostic.context = String(
            StringSlice(unsafe_from_utf8=bytes[line_start:line_end])
        )
        diagnostic.caret_column = column
        return diagnostic^
    fn _error_at(self, offset: Int, message: String) raises:
        raise Error(self._make_diagnostic(offset, message).render())

    fn _advance(mut self) raises:
        self.current = self._scan_next_token()

    fn _expect(mut self, tag: Int, message: String) raises:
        if self.current.tag != tag:
            self._error_at(self.current.start, message)

    fn _scan_next_token(mut self) raises -> JsonToken:
        var bytes = self.text.as_bytes()
        while self.position < len(bytes) and _is_whitespace(bytes[self.position]):
            self.position += 1

        var token = JsonToken()
        token.start = self.position
        token.value_start = self.position
        token.value_end = self.position

        if self.position >= len(bytes):
            token.tag = JsonTokenTag.EOF
            token.end = self.position
            return token^

        var ch = bytes[self.position]
        if ch == Byte(ord("{")):
            token.tag = JsonTokenTag.LeftBrace
            token.end = self.position + 1
            self.position += 1
            return token^
        if ch == Byte(ord("}")):
            token.tag = JsonTokenTag.RightBrace
            token.end = self.position + 1
            self.position += 1
            return token^
        if ch == Byte(ord("[")):
            token.tag = JsonTokenTag.LeftBracket
            token.end = self.position + 1
            self.position += 1
            return token^
        if ch == Byte(ord("]")):
            token.tag = JsonTokenTag.RightBracket
            token.end = self.position + 1
            self.position += 1
            return token^
        if ch == Byte(ord(":")):
            token.tag = JsonTokenTag.Colon
            token.end = self.position + 1
            self.position += 1
            return token^
        if ch == Byte(ord(",")):
            token.tag = JsonTokenTag.Comma
            token.end = self.position + 1
            self.position += 1
            return token^
        if ch == Byte(ord("\"")):
            return self._scan_string_token()
        if ch == Byte(ord("-")) or _is_digit(ch):
            return self._scan_number_token()
        if ch == Byte(ord("t")):
            return self._scan_literal_token("true", JsonTokenTag.TrueLit)
        if ch == Byte(ord("f")):
            return self._scan_literal_token("false", JsonTokenTag.FalseLit)
        if ch == Byte(ord("n")):
            return self._scan_literal_token("null", JsonTokenTag.NullLit)

        self._error_at(self.position, "unexpected token")
        return token^

    fn _scan_literal_token(mut self, literal: StringLiteral, tag: Int) raises -> JsonToken:
        var bytes = self.text.as_bytes()
        var token = JsonToken()
        token.tag = tag
        token.start = self.position
        token.value_start = self.position
        for i in range(len(literal)):
            var idx = self.position + i
            if idx >= len(bytes) or bytes[idx] != Byte(ord(literal[i : i + 1])):
                self._error_at(self.position, "invalid literal")
            token.value_end = idx + 1
        token.end = self.position + len(literal)
        self.position = token.end
        return token^

    fn _scan_string_token(mut self) raises -> JsonToken:
        var bytes = self.text.as_bytes()
        var token = JsonToken()
        token.tag = JsonTokenTag.String
        token.start = self.position
        token.value_start = self.position + 1

        var i = self.position + 1
        while i < len(bytes):
            var ch = bytes[i]
            if ch == Byte(ord("\"")):
                token.end = i + 1
                token.value_end = i
                self.position = i + 1
                return token^
            if ch == Byte(ord("\\")):
                token.has_escape = True
                i += 1
                if i >= len(bytes):
                    self._error_at(token.start, "unterminated string escape")
                var escape = bytes[i]
                if (
                    escape == Byte(ord("\""))
                    or escape == Byte(ord("\\"))
                    or escape == Byte(ord("/"))
                    or escape == Byte(ord("b"))
                    or escape == Byte(ord("f"))
                    or escape == Byte(ord("n"))
                    or escape == Byte(ord("r"))
                    or escape == Byte(ord("t"))
                ):
                    i += 1
                    continue
                if escape == Byte(ord("u")):
                    for j in range(4):
                        var hex_idx = i + 1 + j
                        if hex_idx >= len(bytes) or not _is_hex_digit(bytes[hex_idx]):
                            self._error_at(i, "invalid unicode escape")
                    i += 5
                    continue
                self._error_at(i, "invalid escape sequence")
            if ch < Byte(0x20):
                self._error_at(i, "unescaped control character in string")
            i += 1

        self._error_at(token.start, "unterminated string")
        return token^

    fn _scan_number_token(mut self) raises -> JsonToken:
        var bytes = self.text.as_bytes()
        var token = JsonToken()
        token.tag = JsonTokenTag.Number
        token.start = self.position
        token.value_start = self.position
        token.number_is_integer = True

        var i = self.position
        if bytes[i] == Byte(ord("-")):
            i += 1
            if i >= len(bytes):
                self._error_at(token.start, "number is missing digits")
        if i >= len(bytes):
            self._error_at(token.start, "number is missing digits")
        if bytes[i] == Byte(ord("0")):
            i += 1
            if i < len(bytes) and _is_digit(bytes[i]):
                self._error_at(i, "numbers cannot have leading zeroes")
        elif _is_digit(bytes[i]):
            while i < len(bytes) and _is_digit(bytes[i]):
                i += 1
        else:
            self._error_at(i, "number is missing digits")

        if i < len(bytes) and bytes[i] == Byte(ord(".")):
            token.number_is_integer = False
            i += 1
            if i >= len(bytes) or not _is_digit(bytes[i]):
                self._error_at(i, "fractional number is missing digits")
            while i < len(bytes) and _is_digit(bytes[i]):
                i += 1

        if i < len(bytes) and (bytes[i] == Byte(ord("e")) or bytes[i] == Byte(ord("E"))):
            token.number_is_integer = False
            i += 1
            if i < len(bytes) and (bytes[i] == Byte(ord("+")) or bytes[i] == Byte(ord("-"))):
                i += 1
            if i >= len(bytes) or not _is_digit(bytes[i]):
                self._error_at(i, "exponent is missing digits")
            while i < len(bytes) and _is_digit(bytes[i]):
                i += 1

        token.end = i
        token.value_end = i
        self.position = i
        return token^

    fn _decode_unicode_scalar(
        self, token: JsonToken, start: Int, mut next_index: Int
    ) raises -> (Int, Int):
        var bytes = self.text.as_bytes()
        var value = 0
        for i in range(4):
            value = value * 16 + _hex_value(bytes[start + i])

        next_index = start + 4
        if value >= 0xD800 and value <= 0xDBFF:
            if (
                next_index + 6 > token.value_end
                or bytes[next_index] != Byte(ord("\\"))
                or bytes[next_index + 1] != Byte(ord("u"))
            ):
                self._error_at(start - 2, "high surrogate must be followed by low surrogate")
            var low = 0
            for i in range(4):
                low = low * 16 + _hex_value(bytes[next_index + 2 + i])
            if low < 0xDC00 or low > 0xDFFF:
                self._error_at(next_index + 2, "invalid low surrogate")
            value = 0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00)
            next_index += 6
            return (value, next_index)
        if value >= 0xDC00 and value <= 0xDFFF:
            self._error_at(start - 2, "low surrogate without preceding high surrogate")
        return (value, next_index)

    fn _decode_string(self, token: JsonToken) raises -> String:
        if not token.has_escape:
            return String(
                StringSlice(unsafe_from_utf8=self.text.as_bytes()[token.value_start:token.value_end])
            )

        var decoded = String()
        var i = token.value_start
        var segment_start = i
        var bytes = self.text.as_bytes()
        while i < token.value_end:
            if bytes[i] != Byte(ord("\\")):
                i += 1
                continue
            _append_utf8_segment(decoded, self.text, segment_start, i)
            var escape = bytes[i + 1]
            if escape == Byte(ord("\"")):
                decoded += "\""
                i += 2
            elif escape == Byte(ord("\\")):
                decoded += "\\"
                i += 2
            elif escape == Byte(ord("/")):
                decoded += "/"
                i += 2
            elif escape == Byte(ord("b")):
                decoded += "\b"
                i += 2
            elif escape == Byte(ord("f")):
                decoded += "\f"
                i += 2
            elif escape == Byte(ord("n")):
                decoded += "\n"
                i += 2
            elif escape == Byte(ord("r")):
                decoded += "\r"
                i += 2
            elif escape == Byte(ord("t")):
                decoded += "\t"
                i += 2
            else:
                var (codepoint, next_index) = self._decode_unicode_scalar(token, i + 2, i)
                decoded += String(
                    Codepoint(unsafe_unchecked_codepoint=UInt32(codepoint))
                )
                i = next_index
            segment_start = i
        _append_utf8_segment(decoded, self.text, segment_start, token.value_end)
        return decoded^

    fn _parse_number_node(self, token: JsonToken) raises -> JsonNode:
        var node = JsonNode()
        node.tag = JsonValueTag.Number
        node.start = token.start
        node.end = token.end
        node.number_is_integer = token.number_is_integer
        node.text_value = String(
            StringSlice(unsafe_from_utf8=self.text.as_bytes()[token.value_start:token.value_end])
        )
        node.number_value = atof(StringSlice(node.text_value))
        return node^

    fn _parse_value(mut self, mut doc: JsonDocument) raises -> Int:
        if self.current.tag == JsonTokenTag.String:
            var token = self.current
            self._advance()
            var node = JsonNode()
            node.tag = JsonValueTag.String
            node.start = token.start
            node.end = token.end
            node.text_value = self._decode_string(token)
            return doc.add_node(node)

        if self.current.tag == JsonTokenTag.Number:
            var token = self.current
            self._advance()
            return doc.add_node(self._parse_number_node(token))

        if self.current.tag == JsonTokenTag.TrueLit:
            var token = self.current
            self._advance()
            var node = JsonNode()
            node.tag = JsonValueTag.Bool
            node.start = token.start
            node.end = token.end
            node.bool_value = True
            return doc.add_node(node)

        if self.current.tag == JsonTokenTag.FalseLit:
            var token = self.current
            self._advance()
            var node = JsonNode()
            node.tag = JsonValueTag.Bool
            node.start = token.start
            node.end = token.end
            node.bool_value = False
            return doc.add_node(node)

        if self.current.tag == JsonTokenTag.NullLit:
            var token = self.current
            self._advance()
            var node = JsonNode()
            node.tag = JsonValueTag.Null
            node.start = token.start
            node.end = token.end
            return doc.add_node(node)

        if self.current.tag == JsonTokenTag.LeftBracket:
            return self._parse_array(doc)

        if self.current.tag == JsonTokenTag.LeftBrace:
            return self._parse_object(doc)

        self._error_at(self.current.start, "expected JSON value")
        return -1

    fn _parse_array(mut self, mut doc: JsonDocument) raises -> Int:
        var start = self.current.start
        self._advance()

        if self.current.tag == JsonTokenTag.RightBracket:
            var end = self.current.end
            self._advance()
            var node = JsonNode()
            node.tag = JsonValueTag.Array
            node.start = start
            node.end = end
            node.child_offset = len(doc.array_items)
            node.child_count = 0
            return doc.add_node(node)

        var direct_items = List[Int]()
        while self.current.tag != JsonTokenTag.Invalid:
            direct_items.append(self._parse_value(doc))
            if self.current.tag == JsonTokenTag.Comma:
                self._advance()
                if self.current.tag == JsonTokenTag.RightBracket:
                    self._error_at(self.current.start, "trailing comma in array")
                continue
            if self.current.tag == JsonTokenTag.RightBracket:
                var end = self.current.end
                self._advance()
                var child_offset = len(doc.array_items)
                for i in range(len(direct_items)):
                    doc.array_items.append(direct_items[i])
                var node = JsonNode()
                node.tag = JsonValueTag.Array
                node.start = start
                node.end = end
                node.child_offset = child_offset
                node.child_count = len(direct_items)
                return doc.add_node(node)
            self._error_at(self.current.start, "expected ',' or ']' after array element")

        self._error_at(self.current.start, "invalid array state")
        return -1

    fn _parse_object(mut self, mut doc: JsonDocument) raises -> Int:
        var start = self.current.start
        self._advance()

        if self.current.tag == JsonTokenTag.RightBrace:
            var end = self.current.end
            self._advance()
            var node = JsonNode()
            node.tag = JsonValueTag.Object
            node.start = start
            node.end = end
            node.child_offset = len(doc.object_entries)
            node.child_count = 0
            return doc.add_node(node)

        var direct_entries = List[JsonObjectEntry]()
        while self.current.tag != JsonTokenTag.Invalid:
            self._expect(JsonTokenTag.String, "expected object key string")
            var key_token = self.current
            var key = self._decode_string(key_token)
            self._advance()
            self._expect(JsonTokenTag.Colon, "expected ':' after object key")
            self._advance()
            var value_index = self._parse_value(doc)
            direct_entries.append(
                JsonObjectEntry(key, value_index, key_token.start)
            )
            if self.current.tag == JsonTokenTag.Comma:
                self._advance()
                if self.current.tag == JsonTokenTag.RightBrace:
                    self._error_at(self.current.start, "trailing comma in object")
                continue
            if self.current.tag == JsonTokenTag.RightBrace:
                var end = self.current.end
                self._advance()
                var child_offset = len(doc.object_entries)
                for i in range(len(direct_entries)):
                    doc.object_entries.append(direct_entries[i])
                var node = JsonNode()
                node.tag = JsonValueTag.Object
                node.start = start
                node.end = end
                node.child_offset = child_offset
                node.child_count = len(direct_entries)
                return doc.add_node(node)
            self._error_at(self.current.start, "expected ',' or '}' after object entry")

        self._error_at(self.current.start, "invalid object state")
        return -1

    fn parse(mut self) raises -> JsonDocument:
        var doc = JsonDocument(self.text)
        self._advance()
        doc.root_index = self._parse_value(doc)
        if self.current.tag != JsonTokenTag.EOF:
            self._error_at(self.current.start, "unexpected trailing content")
        return doc^
fn parse_json_native(text: String) raises -> JsonDocument:
    var owned_text = String(text)
    var parser = JsonParser(owned_text)
    return parser.parse()


fn load_json_native(path: String) raises -> JsonDocument:
    return parse_json_native(Path(path).read_text())
