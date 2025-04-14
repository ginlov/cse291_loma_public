import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW1: TODO
            # Mutate arguments
            new_args = [loma_ir.Arg(
                arg.id,
                autodiff.type_to_diff_type(diff_structs, arg.t),
                arg.i
            ) for arg in node.args]

            # New function
            new_node = loma_ir.FunctionDef(
                diff_func_id,
                new_args,
                irmutator.flatten([self.mutate_stmt(stmt) for stmt in node.body]),
                node.is_simd,
                autodiff.type_to_diff_type(diff_structs, node.ret_type),
                lineno=node.lineno
            )
            return new_node
        
        def mutate_return(self, node):
            # HW1: TODO
            return super().mutate_return(node)

        def mutate_declare(self, node):
            # HW1: TODO
            return loma_ir.Declare(
                node.target,
                autodiff.type_to_diff_type(diff_structs, node.t),
                self.mutate_expr(node.val) if node.val is not None else None,
                lineno=node.lineno
            )

        def mutate_assign(self, node):
            # HW1: TODO
            return loma_ir.Assign(
                self.mutate_expr(node.target),
                self.mutate_expr(node.val)
            )

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW1: TODO
            return loma_ir.Call('make__dfloat', [node, loma_ir.ConstFloat(0.0)])
        
        def mutate_const_int(self, node):
            # HW1: TODO
            return super().mutate_const_int(node)

        def mutate_var(self, node):
            return super().mutate_var(node)

        def mutate_array_access(self, node):
            # HW1: TODO
            return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW1: TODO
            struct = self.mutate_expr(node.struct)
            match struct:
                case loma_ir.Call():
                    match struct.id:
                        case 'int2float':
                            if node.member_id == 'val':
                                return struct.args[0]
                            else:
                                return loma_ir.ConstFloat(0.0)
                case loma_ir.ConstInt():
                    if node.member_id == 'val':
                        return loma_ir.Call('int2float', [struct])
                    else:
                        return loma_ir.ConstFloat(0.0)
            return super().mutate_struct_access(node)

        def mutate_add(self, node):
            # HW1: TODO
            left = node.left
            right = node.right
            left_val = self.mutate_struct_access(loma_ir.StructAccess(left, 'val'))
            left_dval = self.mutate_struct_access(loma_ir.StructAccess(left, 'dval'))
            right_val = self.mutate_struct_access(loma_ir.StructAccess(right, 'val'))
            right_dval = self.mutate_struct_access(loma_ir.StructAccess(right, 'dval'))
            ret_val = loma_ir.BinaryOp(
                loma_ir.Add(),
                left_val,
                right_val,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval = loma_ir.BinaryOp(
                loma_ir.Add(),
                left_dval,
                right_dval,
                lineno = node.lineno,
                t = loma_ir.Float())
            return loma_ir.Call('make__dfloat', [ret_val, ret_dval])

        def mutate_sub(self, node):
            # HW1: TODO
            left = node.left
            right = node.right
            left_val = self.mutate_struct_access(loma_ir.StructAccess(left, 'val'))
            left_dval = self.mutate_struct_access(loma_ir.StructAccess(left, 'dval'))
            right_val = self.mutate_struct_access(loma_ir.StructAccess(right, 'val'))
            right_dval = self.mutate_struct_access(loma_ir.StructAccess(right, 'dval'))
            ret_val = loma_ir.BinaryOp(
                loma_ir.Sub(),
                left_val,
                right_val,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval = loma_ir.BinaryOp(
                loma_ir.Sub(),
                left_dval,
                right_dval,
                lineno = node.lineno,
                t = loma_ir.Float())
            return loma_ir.Call('make__dfloat', [ret_val, ret_dval])

        def mutate_mul(self, node):
            # HW1: TODO
            left = node.left
            right = node.right
            left_val = self.mutate_struct_access(loma_ir.StructAccess(left, 'val'))
            left_dval = self.mutate_struct_access(loma_ir.StructAccess(left, 'dval'))
            right_val = self.mutate_struct_access(loma_ir.StructAccess(right, 'val'))
            right_dval = self.mutate_struct_access(loma_ir.StructAccess(right, 'dval'))
            ret_val = loma_ir.BinaryOp(
                loma_ir.Mul(),
                left_val,
                right_val,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval1 = loma_ir.BinaryOp(
                loma_ir.Mul(),
                left_dval,
                right_val,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval2 = loma_ir.BinaryOp(
                loma_ir.Mul(),
                left_val,
                right_dval,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval = loma_ir.BinaryOp(
                loma_ir.Add(),
                ret_dval1,
                ret_dval2,
                lineno=node.lineno,
                t=loma_ir.Float()
            )
            return loma_ir.Call('make__dfloat', [ret_val, ret_dval])

        def mutate_div(self, node):
            # HW1: TODO
            left = node.left
            right = node.right
            left_val = self.mutate_struct_access(loma_ir.StructAccess(left, 'val'))
            left_dval = self.mutate_struct_access(loma_ir.StructAccess(left, 'dval'))
            right_val = self.mutate_struct_access(loma_ir.StructAccess(right, 'val'))
            right_dval = self.mutate_struct_access(loma_ir.StructAccess(right, 'dval'))
            ret_val = loma_ir.BinaryOp(
                loma_ir.Div(),
                left_val,
                right_val,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval1 = loma_ir.BinaryOp(
                loma_ir.Div(),
                left_dval,
                right_val,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval2 = loma_ir.BinaryOp(
                loma_ir.Div(),
                right_dval,
                right_val,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval3 = loma_ir.BinaryOp(
                loma_ir.Mul(),
                ret_val,
                ret_dval2,
                lineno = node.lineno,
                t = loma_ir.Float())
            ret_dval = loma_ir.BinaryOp(
                loma_ir.Sub(),
                ret_dval1,
                ret_dval3,
                lineno=node.lineno,
                t=loma_ir.Float()
            )
            return loma_ir.Call('make__dfloat', [ret_val, ret_dval])

        def mutate_call(self, node):
            # HW1: TODO
            match node.id:
                case 'sin':
                    angle_val = loma_ir.StructAccess(node.args[0], 'val')
                    angle_dval = loma_ir.StructAccess(node.args[0], 'dval')
                    ret_val = loma_ir.Call('sin', [angle_val])
                    ret_dval1 = loma_ir.Call('cos', [angle_val])
                    ret_dval2 = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        angle_dval,
                        ret_dval1,
                        lineno=node.lineno,
                        t=loma_ir.Float())
                    return loma_ir.Call('make__dfloat', [ret_val, ret_dval2])
                case 'cos':
                    angle_val = loma_ir.StructAccess(node.args[0], 'val')
                    angle_dval = loma_ir.StructAccess(node.args[0], 'dval')
                    ret_val = loma_ir.Call('cos', [angle_val])
                    ret_dval1 = loma_ir.Call('sin', [angle_val])
                    ret_dval2 = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        angle_dval,
                        ret_dval1,
                        lineno=node.lineno,
                        t=loma_ir.Float())
                    ret_dval3 = loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        ret_dval2,
                        lineno=node.lineno,
                        t=loma_ir.Float()
                    )
                    return loma_ir.Call('make__dfloat', [ret_val, ret_dval3])
                case 'sqrt':
                    in_val = loma_ir.StructAccess(node.args[0], 'val')
                    in_dval = loma_ir.StructAccess(node.args[0], 'dval')
                    ret_val = loma_ir.Call('sqrt', [in_val])
                    ret_dval1 = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        in_dval,
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.ConstFloat(2.0),
                            ret_val,
                            lineno=node.lineno,
                            t=loma_ir.Float()),
                        lineno=node.lineno,
                        t=loma_ir.Float())
                    return loma_ir.Call('make__dfloat', [ret_val, ret_dval1])
                case 'pow':
                    base_val = loma_ir.StructAccess(node.args[0], 'val')
                    base_dval = loma_ir.StructAccess(node.args[0], 'dval')
                    exp_val = loma_ir.StructAccess(node.args[1], 'val')
                    exp_dval = loma_ir.StructAccess(node.args[1], 'dval')
                    ret_val = loma_ir.Call('pow', [base_val, exp_val])
                    ret_dval1 = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        base_dval,
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            ret_val,
                            loma_ir.BinaryOp(
                                loma_ir.Div(),
                                exp_val,
                                base_val,
                                lineno=node.lineno,
                                t=loma_ir.Float()
                            ),
                            lineno=node.lineno,
                            t=loma_ir.Float()
                        ),
                        lineno=node.lineno,
                        t=loma_ir.Float()
                    )

                    ret_dval2 = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        exp_dval,
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            ret_val,
                            loma_ir.Call('log', [base_val]),
                            lineno=node.lineno,
                            t=loma_ir.Float()
                        ),
                        lineno=node.lineno,
                        t=loma_ir.Float()
                    )

                    ret_dval3 = loma_ir.BinaryOp(
                            loma_ir.Add(),
                            ret_dval1,
                            ret_dval2,
                            lineno=node.lineno,
                            t=loma_ir.Float()
                        )
                    return loma_ir.Call('make__dfloat', [ret_val, ret_dval3])
                case 'exp':
                    in_val = loma_ir.StructAccess(node.args[0], 'val')
                    in_dval = loma_ir.StructAccess(node.args[0], 'dval')
                    ret_val = loma_ir.Call('exp', [in_val])
                    ret_dval1 = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        in_dval,
                        ret_val,

                        lineno=node.lineno,
                        t=loma_ir.Float())
                    return loma_ir.Call('make__dfloat', [ret_val, ret_dval1])
                case 'log':
                    in_val = loma_ir.StructAccess(node.args[0], 'val')
                    in_dval = loma_ir.StructAccess(node.args[0], 'dval')
                    ret_val = loma_ir.Call('log', [in_val])
                    ret_dval1 = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        in_dval,
                        in_val,
                        lineno=node.lineno,
                        t=loma_ir.Float())
                    return loma_ir.Call('make__dfloat', [ret_val, ret_dval1])
                case 'int2float':
                    return loma_ir.Call('make__dfloat', [node.args[0], loma_ir.ConstFloat(0.0)])
                case 'float2int':
                    arg = self.mutate_struct_access(loma_ir.StructAccess(node.args[0], 'val'))
                    return loma_ir.Call('float2int', [arg])
                case _:
                    return super().mutate_call(node)

    return FwdDiffMutator().mutate_function_def(func)
