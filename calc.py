# Token types
EOF = 'EOF'
INTEGER = 'INTEGER'
PLUS = 'PLUS'
MINUS = 'MINUS'

class Token:
    def __init__(self, type, value):
        # token type: INTEGER, PLUS, or EOF
        self.type = type
        # token value: integer (0, 1, 2, ..., 9), '+', or None
        self.value = value

    def __str__(self):
        return f'Token({self.type}, {repr(self.value)})'

    def __repr__(self):
        return self.__str__()

class Interpreter:
    def __init__(self, text):
        # The input expression, e.g., "3+5"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        # current token instance
        self.current_token = None
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Error parsing input')

    def get_next_token(self):
        """Lexical analyzer (scanner or tokenizer)
        
        This method is responsible for breaking a sentence
        apart into tokens.
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
        
            self.error()
            
        return Token(EOF, None)

    def eat(self, token_type):
        """Compare the current token type with the passed token type and if they match, eat the current token"""
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error()

    
    
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
            
    def advance(self):
        """Advance the 'pos' pointer and set the 'current_char' variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]
            
    def expr(self):
        """Parser / Interpreter
        expr -> INTEGER PLUS INTEGER
        expr -> INTEGER MINUNS INTEGER 
        """
        # set current token to the first token taken from the input
        self.current_token = self.get_next_token()
        
        # we expect the current token to be an integer
        left = self.current_token
        self.eat(INTEGER)
        
        # we expect the current token to be either a '+' or '-'
        op = self.current_token
        if op.type == PLUS:
            self.eat(PLUS)
        else:
            self.eat(MINUS)
            
        # we expect the current token to be an integer
        right = self.current_token
        self.eat(INTEGER)
        # after the above call the self.current_token is set to 
        #EOF token 
        
        #at this point either the INTEGER  PLUS INTEGER  or
        #the INTEGER MINUS INTEGER sequence of tokens
        #has been successfully found and the method can just 
        #return the result of adding or subtracting two integers,
        #thus effectively interpreting client input
        if op.type == PLUS:
            result = left.value + right.value
        else:
            result = left.value - right.value
        return result
        
        
def main():
    while True:
        try:
            text = input('calc> ')
        except EOFError:
            break
        if not text:
            continue
        interpreter = Interpreter(text)
        result = interpreter.expr()
        print(result)

if __name__ == '__main__':
    main()
