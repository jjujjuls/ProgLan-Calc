#Token types
#
#EOF (end-of-file) token is used to indicate that
#there is no more input left for lexical analysis
INTEGER, PLUS, MINUS, MUL, DIV, EOF = 'INTEGER', 'PLUS', 'MINUS', 'MUL', 'DIV', 'EOF'

class Token(object):
    def __init__(self, type, value):
        # tokentype: INTEGER, MUL, DIV, or EOF
        self.type = type
        # token value: 0,1,2,3,4,5,6,7,8,9,+, or None
        # token value: non-negative integer value, '*', '/', '+', or None
        self.value = value

    def __str__(self):
        """String representation of the class instance.
        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()

class Lexer(object):

    def __init__(self, text):
        #client string input, e.g. "3 * 5", "12 / 3 * 4", etc
        self.text = text
        #self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        """Advance the 'pos' pointer and set the 'current_char' variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None #Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):
        """Return a (multidigit) integer consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')
            
            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/')

            self.error()

        return Token(EOF, None)

    def term(self):
        """term : factor ((MUL | DIV) factor)*"""
        result = self.factor()

        while self.current_token.type in (MUL, DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
                result = result * self.factor()
            elif token.type == DIV:
                self.eat(DIV)
                result = result / self.factor()

        return result
    
class Interpreter(object):
    def __init__(self, lexer):
        # client string input, e.g. "3+5"
        self.lexer = lexer
        # self.pos is an index into self.text
        # self.pos = 0
        # # current token instance
        # self.current_token = None
        # self.current_char = self.text[self.pos]
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def factor(self):
        """Return an INTEGER token value"""
        token = self.current_token
        self.eat(INTEGER)
        return token.value

    # def advance(self):
    #     """Advance the 'pos' pointer and set 'current_char' variable."""
    #     self.pos += 1
    #     if self.pos > len(self.text) - 1:
    #         self.current_char = None #Indicates end of input
    #     else:
    #         self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid Syntax')

    # def get_next_token(self):
    #     """Lexical analyzer (also known as scanner or tokenizer)

    #     This method is responsible for breaking a sentence
    #     apart into tokens. One token at a time.
    #     """
    #     while self.current_char is not None:
    #         if self.current_char.isspace():
    #             self.skip_whitespace()
    #             continue

    #         if self.current_char.isdigit():
    #             return Token(INTEGER, self.integer())

    #         if self.current_char == '*':
    #             self.advance()
    #             return Token(MUL, '*')

    #         if self.current_char == '/':
    #             self.advance()
    #             return Token(DIV, '/')

    #         self.error()

    #     return Token(EOF, None)
        # text = self.text

        # while self.pos < len(text) and text[self.pos].isspace():
        #     self.pos += 1

        # is self.pos index past the end of the self.text ?
        # if so, then return EOF token because there is no more
        # input left to convert into tokens

        # if self.pos > len(text) - 1:
        #     return Token(EOF, None)

        #get a character at the position self.pos and decide
        #what token to create based on the single character
        # current_char = text[self.pos]

        #if the character is a digit then convert it to
        #integer, create an INTEGER token, increment self.pos
        #index to point to the next character after the digit,
        #and return the INTEGER token
        # if current_char.isdigit():
        #     start_pos = self.pos
        #     while self.pos < len(text) and text[self.pos].isdigit():
        #         self.pos += 1
        #     token = Token(INTEGER, int(text[start_pos:self.pos]))
        #     return token

        # if current_char == '+':
        #     self.pos += 1
        #     return Token(PLUS, current_char)

        # self.error()

    # def skip_whitespace(self):
    #     while self.current_char is not None and self.current_char.isspace():
    #         self.advance()

    # def integer(self):
    #     """Return a (multidigit) integer consumed from the input."""
    #     result = ''
    #     while self.current_char is not None and self.current_char.isdigit():
    #         result += self.current_char
    #         self.advance()
    #     return int(result)

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    # def term(self):
    #     """Return an INTEGER token value"""
    #     token =self.current_token
    #     self.eat(INTEGER)
    #     return token.value

    def expr(self):
        """Arithmetic expression parser / interpreter.

        calc> 14 + 2 * 3 - 6 / 2
        17

        expr : term ((PLUS | MINUS) term)*
        term : factor ((MUL | DIV) factor)*
        factor : INTEGER
        """
        result = self.term()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
                result = result * self.term()
            elif token.type == MINUS:
                self.eat(MINUS)
                result = result // self.term()
                
        return result
        # set current token to the first token taken from the input
        # self.current_token = self.get_next_token()

        # result = self.term()
        # while self.current_token.type in (PLUS, MINUS):
        #     token = self.current_token
        #     if token.type == PLUS:
        #         self.eat(PLUS)
        #         result = result + self.term()
        #     elif token.type == MINUS:
        #         self.eat(MINUS)
        #         result = result - self.term()
        # return result
        # expect the current token to be a single-digit integer
        # left = self.current_token
        # self.eat(INTEGER)

        # expect the current token to be a '+' token
        # op = self.current_token
        # if op.type == PLUS:
        #     self.eat(PLUS)
        # else:
        #     self.eat(MINUS)

        # # expect the current token to be a single-digit integer
        # right = self.current_token
        # self.eat(INTEGER)
        # # after the above call the self.current_token is set to
        # EOF token

        # at this point INTEGER PLUS INTEGER sequence of tokens
        # has been successfully found and the method can just
        # return the result of adding two integers, thus
        # effectively interpreting client input
        # if op.type == PLUS:
        #     result = left.value + right.value
        # else:
        #     result = left.value - right.value
        # return result

def main():
    while True:
        try:
            # To run under Python3 replace 'raw_input' call
            # with 'input'
            text = input('calc> ')
        except EOFError:
            break
        if not text:
            continue
        lexer = Lexer(text)
        interpreter = Interpreter(lexer)
        result = interpreter.expr()
        print(result)

if _name_ == '_main_':
    main()
