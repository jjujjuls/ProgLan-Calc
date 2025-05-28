import sys

(INTEGER, REAL, INTEGER_CONST, REAL_CONST, PLUS, MINUS, MUL, INTEGER_DIV, FLOAT_DIV, LPAREN, RPAREN, ID, ASSIGN, BEGIN, END, SEMI, DOT, PROGRAM, VAR, COLON, COMMA, EOF) = (
    'INTEGER', 'REAL', 'INTEGER_CONST', 'REAL_CONST', 'PLUS', 'MINUS', 'MUL', 'INTEGER_DIV', 'FLOAT_DIV', '(', ')', 'ID', 'ASSIGN', 'BEGIN', 'END', 'SEMI', 'DOT', 'PROGRAM', 'VAR', ':', ',', 'EOF'
)

class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return f'Token({self.type}, {repr(self.value)})'

    def __repr__(self):
        return self.__str__()

RESERVED_KEYWORDS = {
    'PROGRAM': Token(PROGRAM, PROGRAM),
    'VAR': Token(VAR, VAR),
    'DIV': Token(INTEGER_DIV, 'DIV'),
    'INTEGER': Token(INTEGER, INTEGER),
    'REAL': Token(REAL, REAL),
    'BEGIN': Token(BEGIN, BEGIN),
    'END': Token(END, END),
}

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        if self.pos + 1 < len(self.text):
            return self.text[self.pos + 1]
        return None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
            return Token(REAL_CONST, float(result))
        else:
            return Token(INTEGER_CONST, int(result))

    def _id(self):
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        token = RESERVED_KEYWORDS.get(result.upper(), Token(ID, result))
        return token

    def get_next_token(self):
        while self.current_char is not None:

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()

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
                return Token(FLOAT_DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            if self.current_char == ':' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(ASSIGN, ':=')

            if self.current_char == ':':
                self.advance()
                return Token(COLON, ':')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            if self.current_char == '{':
                while self.current_char != '}' and self.current_char is not None:
                    self.advance()
                self.advance()  
                continue

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            raise Exception(f"Invalid character: '{self.current_char}' at position {self.pos}")

        return Token(EOF, None)

class AST:
    pass

class Program(AST):
    def __init__(self, name, block):
        self.name = name
        self.block = block

class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement

class VarDecl(AST):
    def __init__(self, var_nodes, type_node):
        self.var_nodes = var_nodes  # list of Var nodes
        self.type_node = type_node

class Type(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.type

class Compound(AST):
    def __init__(self):
        self.children = []

class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class Var(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class NoOp(AST):
    pass

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.op = op
        self.expr = expr

class Num(AST):
    def __init__(self, token):
        self.token = token
        if token.type == INTEGER_CONST:
            self.value = int(token.value)
        elif token.type == REAL_CONST:
            self.value = float(token.value)

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        self.eat(PROGRAM)
        var = self.variable()
        prog_name = var.value
        self.eat(SEMI)
        block = self.block()
        self.eat(DOT)
        return Program(prog_name, block)

    def block(self):
        declarations = self.declarations()
        compound_statement = self.compound_statement()
        return Block(declarations, compound_statement)

    def declarations(self):
        declarations = []
        if self.current_token.type == VAR:
            self.eat(VAR)
            while self.current_token.type == ID:
                var_decl = self.variable_declaration()
                declarations.extend(var_decl)
                self.eat(SEMI)
        return declarations

    def variable_declaration(self):
        var_nodes = [self.variable()]
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            var_nodes.append(self.variable())
        self.eat(COLON)
        type_node = self.type_spec()
        return [VarDecl(var_nodes, type_node)]

    def type_spec(self):
        token = self.current_token
        if token.type == INTEGER:
            self.eat(INTEGER)
        elif token.type == REAL:
            self.eat(REAL)
        else:
            self.error()
        return Type(token)

    def compound_statement(self):
        if self.current_token.type == BEGIN:
            self.eat(BEGIN)
            nodes = self.statement_list()
            self.eat(END)
        else:
            nodes = [self.statement()]
        compound_node = Compound()
        compound_node.children = nodes
        return compound_node

    def statement_list(self):
        node = self.statement()
        nodes = [node]
        while self.current_token.type == SEMI:
            self.eat(SEMI)
            nodes.append(self.statement())
        return nodes

    def statement(self):
        if self.current_token.type == BEGIN:
            return self.compound_statement()
        elif self.current_token.type == ID:
            return self.assignment_statement()
        else:
            return NoOp()

    def assignment_statement(self):
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        right = self.expr()
        return Assign(left, token, right)

    def variable(self):
        node = Var(self.current_token)
        self.eat(ID)
        return node

    def expr(self):
        node = self.term()
        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)
            node = BinOp(node, token, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.current_token.type in (MUL, INTEGER_DIV, FLOAT_DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == INTEGER_DIV:
                self.eat(INTEGER_DIV)
            elif token.type == FLOAT_DIV:
                self.eat(FLOAT_DIV)
            node = BinOp(node, token, self.factor())
        return node

    def factor(self):
        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            return UnaryOp(token, self.factor())
        elif token.type == MINUS:
            self.eat(MINUS)
            return UnaryOp(token, self.factor())
        elif token.type == INTEGER_CONST:
            self.eat(INTEGER_CONST)
            return Num(token)
        elif token.type == REAL_CONST:
            self.eat(REAL_CONST)
            return Num(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        elif token.type == ID:
            return self.variable()
        else:
            self.error()

    def parse(self):
        node = self.program()
        if self.current_token.type != EOF:
            self.error()
        return node

class Symbol:
    def __init__(self, name, type=None):
        self.name = name
        self.type = type

class BuiltinTypeSymbol(Symbol):
    def __init__(self, name):
        super().__init__(name)

    def __str__(self):
        return self.name

class VarSymbol(Symbol):
    def __init__(self, name, type):
        super().__init__(name, type)

    def __str__(self):
        return f"<{self.name}:{self.type}>"

class SymbolTable:
    def __init__(self):
        self._symbols = {}

    def define(self, symbol):
        self._symbols[symbol.name] = symbol

    def lookup(self, name):
        return self._symbols.get(name)

class SymbolTabBuilder:
    def __init__(self):
        self.symtab = SymbolTable()

    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")

    def visit_Program(self, node):
        self.visit(node.block)

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_VarDecl(self, node):
        type_name = node.type_node.token.type
        if type_name == "INTEGER":
            type_symbol = BuiltinTypeSymbol("INTEGER")
        elif type_name == "REAL":
            type_symbol = BuiltinTypeSymbol("REAL")
        else:
            return

        for var_node in node.var_nodes:
            symbol = VarSymbol(var_node.value, type_symbol)
            self.symtab.define(symbol)

    def visit_Assign(self, node):
        var_name = node.left.value
        symbol = self.symtab.lookup(var_name)
        if symbol is None:
            raise NameError(f"Variable '{var_name}' is not declared.")
        self.visit(node.right)

    def visit_Var(self, node):
        var_name = node.value
        symbol = self.symtab.lookup(var_name)
        if symbol is None:
            raise NameError(f"Variable '{var_name}' is not declared.")

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)  # This line was missing

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

    def visit_NoOp(self, node):
        pass

    def visit_Num(self, node):
        pass

class Interpreter:
    GLOBAL_MEMORY = {}

    def __init__(self, tree):
        self.tree = tree

    def interpret(self):
        self.GLOBAL_MEMORY = {}
        self.visit(self.tree)

    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")

    def visit_Program(self, node):
        self.visit(node.block)

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        self.visit(node.compound_statement)

    def visit_VarDecl(self, node):
        pass

    def visit_Type(self, node):
        pass

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_Assign(self, node):
        var_name = node.left.value
        value = self.visit(node.right)
        self.GLOBAL_MEMORY[var_name] = value

    def visit_Var(self, node):
        var_name = node.value
        return self.GLOBAL_MEMORY.get(var_name)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if node.op.type == PLUS:
            return left + right
        elif node.op.type == MINUS:
            return left - right
        elif node.op.type == MUL:
            return left * right
        elif node.op.type == INTEGER_DIV:
            return left // right
        elif node.op.type == FLOAT_DIV:
            return left / right

    def visit_UnaryOp(self, node):
        expr_val = self.visit(node.expr)
        if node.op.type == PLUS:
            return +expr_val
        elif node.op.type == MINUS:
            return -expr_val

    def visit_NoOp(self, node):
        pass

    def visit_Num(self, node):
        return node.value

def main():
    import sys
    text = open(sys.argv[1], 'r').read()

    lexer = Lexer(text)
    parser = Parser(lexer)
    tree = parser.parse()
    symtab_builder = SymbolTabBuilder()
    symtab_builder.visit(tree)

    print('')
    print('Symbol Table contents:')
    print(symtab_builder.symtab)

    interpreter = Interpreter(tree)
    result = interpreter.interpret()

    print('')
    print('Run-time GLOBAL_MEMORY contents:')
    for k, v in sorted(interpreter.GLOBAL_MEMORY.items()):
        print('{} = {}'.format(k, v))

if __name__ == '__main__':
    main()
