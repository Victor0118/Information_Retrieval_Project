depth=0
class exprNode(object):
    '''Class of bool expression tree'''
    def __init__(self,value):
        self.op_value=value
        self.leftchild=None
        self.rightchild=None
    def linkNode(self,left,right):
        self.leftchild=left
        self.rightchild=right
        return self
    def printout(self):
        global depth
        leftnode=self.leftchild
        rightnode=self.rightchild
        if leftnode!=None:
            depth=depth-1
            leftnode.printout()
            depth=depth+1
        print self.op_value,depth
        if rightnode!=None:
            depth=depth-1
            rightnode.printout()
            depth=depth+1

        #if leftres==None and rightres==None:
        #    print self.op_value,depth
    def calc(self,index,docs_num):
    # '''calculate the docs indexes according to bool expression tree'''
        leftnode=self.leftchild
        rightnode=self.rightchild
        if self.op_value=='and':

            leftdocs_indices=self.leftchild.calc(index,docs_num)
            rightdocs_indices=self.rightchild.calc(index,docs_num)
            res=set(leftdocs_indices) & set(rightdocs_indices)

            return list(res)
        elif self.op_value=='or':
            leftdocs_indices=self.leftchild.calc(index,docs_num)
            rightdocs_indices=self.rightchild.calc(index,docs_num)
            res=set(leftdocs_indices)| set(rightdocs_indices)
            return list(res)
        elif self.op_value=='not':
            leftdocs_indices=range(docs_num)
            rightdocs_indices=self.rightchild.calc(index,docs_num)
            res=set(leftdocs_indices)- set(rightdocs_indices)

            return list(res)
        else:
            res=index.search_terms(self.op_value.split())
            #print "debug info:",self.op_value.split(),res
            return res
class parser(object):
    '''parse bool expression'''
    def __init__(self,terms):
        self.terms=terms
        self.i=0
        self.token=self.terms[self.i]
    def parse(self):
        return self.exp()
    def match(self,expectedToken):
        #token=self.terms[self.i]
        if self.token==expectedToken:           
            res= exprNode( self.token)
        else:           
            print 'systax error error',token
            exit()
        self.i=self.i+1
        if(self.i<len(self.terms)):
            self.token=self.terms[self.i]
        return res
    def factor(self):
        if(self.token=='not'):
            op=self.match('not')
            right=self.negation()
            return op.linkNode(None, right)
        else:
            return self.negation()
    def negation(self):
        if(self.token=='('):
            self.match('(')
            exp=self.exp()
            self.match(')')
            return exp
        else:
            return self.words()
    def words(self):
        tokenlist=[]
        if(self.token!='(' and self.token !=')' and self.token !='and' and self.token != 'or' and self.token!='not'):
            '''for single word'''
            return self.match(self.token)#one words
        else:
            print "syntax error",self.token
            exit()
            '''for multiple words query
            tokenlist.append(self.token)
            self.match(self.token)
        return exprNode(tokenlist)'''
    def term(self):
        left=self.factor()
        while self.token=='and':
            op=self.match('and' )
            right=self.factor()
            left=op.linkNode(left, right)
        return left
    def exp(self):
        left=self.term()
        while(self.token=='or'):
            op=self.match('or')
            right=self.term()
            left=op.linkNode(left, right)
        return left
