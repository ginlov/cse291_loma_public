import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random

# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

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
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target, deriv, overwrite):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    return [loma_ir.Assign(target,
                        loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_args(lhs, output_args):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.StructAccess():
                return check_lhs_is_output_args(lhs.struct, output_args)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_args(lhs.array, output_args)
            case _:
                assert False

    def check_while_depth(node):
        depth = 1
        for stmt in node.body:
            if isinstance(stmt, loma_ir.While):
                depth += check_while_depth(stmt)
                break
        return depth
    
    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    # Furthermore, it normalizes all Assign statements
    # with a function call
    # z = f(...)
    # into a declaration followed by an assignment
    # _tmp : [z's type]
    # _tmp = f(...)
    # z = _tmp
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(\
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            self.has_call_expr = False
            val = self.mutate_expr(node.val)
            if self.has_call_expr:
                # turn the assignment into a declaration plus
                # an assignment
                self.tmp_count += 1
                tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                self.tmp_count += 1
                self.tmp_declare_stmts.append(loma_ir.Declare(\
                    tmp_name,
                    target.t,
                    lineno = node.lineno))
                tmp_var = loma_ir.Var(tmp_name, t = target.t)
                assign_tmp = loma_ir.Assign(\
                    tmp_var,
                    val,
                    lineno = node.lineno)
                assign_target = loma_ir.Assign(\
                    target,
                    tmp_var,
                    lineno = node.lineno)
                return self.tmp_assign_stmts + [assign_tmp, assign_target]
            else:
                return self.tmp_assign_stmts + [loma_ir.Assign(\
                    target,
                    val,
                    lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

        def mutate_call(self, node):
            self.has_call_expr = True
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.
    class ForwardPassMutator(irmutator.IRMutator):
        def compute_stack_size(self, node):
            float_size = 0
            int_size = 0
            def visit_target(target):
                stack_float_size = 0
                stack_int_size = 0
                match target:
                    case loma_ir.StructAccess():
                        members = target.struct.t.members
                        for mem in members:
                            if mem.id == target.member_id:
                                f, i = visit_target(mem)
                                stack_float_size += f
                                stack_int_size += i
                                break
                        return [stack_float_size, stack_int_size]
                    case _:
                        match target.t:
                            case loma_ir.Float():
                                return [1, 0]
                            case loma_ir.Int():
                                return [0, 1]
                            case loma_ir.Array():
                                for i in range(target.t.static_size):
                                    f, i = visit_target(loma_ir.ArrayAccess(target, i))
                                    stack_float_size += f
                                    stack_int_size += i
                                return [stack_float_size, stack_int_size]
                            case loma_ir.Struct():
                                for i in target.t.members:
                                    f, i = visit_target(i)
                                    stack_float_size += f
                                    stack_int_size += i
                                return [stack_float_size, stack_int_size]
                            case _:
                                return [0, 0]
            for stmt in node:
                match stmt:
                    case loma_ir.Assign():
                        target = stmt.target
                        f, i = visit_target(target)
                        float_size += f
                        int_size += i
                    case loma_ir.IfElse():
                        f, i = self.compute_stack_size(stmt.then_stmts)
                        float_size += f
                        int_size += i
                        f, i = self.compute_stack_size(stmt.else_stmts)
                        float_size += f
                        int_size += i
                    case loma_ir.CallStmt():
                        if stmt.call.id not in ['cos', 'sin', 'low', 'log', 'sqrt', 'exp', 'int2float', 'float2int', 'atomic_add']:
                            last_args = stmt.call.args[-1]
                            match last_args:
                                case loma_ir.ArrayAccess():
                                    last_args = last_args.array
                                case loma_ir.StructAccess():
                                    last_args = last_args.struct
                            if check_lhs_is_output_args(last_args, self.output_args):
                                continue
                            f, i = visit_target(last_args)
                            float_size += f
                            int_size += i
                    case loma_ir.While():
                        f, i = self.compute_stack_size(stmt.body)
                        float_size += f*stmt.max_iter
                        int_size += i*stmt.max_iter
                    case _:
                        continue
            return [float_size, int_size]
                        
        def mutate_body(self, node):
            new_body = []
            self.ret_type = node.ret_type
            self.arg_id_to_diff_id = {}
            self.output_args = [arg.id for arg in node.args if arg.i == loma_ir.Out()]
            stack_float_size, stack_int_size = self.compute_stack_size(node.body)
            
            if stack_float_size > 0:
                stack_float_init = loma_ir.Declare(
                    '_t_float',
                    loma_ir.Array(loma_ir.Float(), stack_float_size),
                )
                pointer_float_init = loma_ir.Declare(
                    '_stack_ptr_float',
                    loma_ir.Int(),
                    loma_ir.ConstInt(0)
                )
                new_body += [stack_float_init, pointer_float_init]
            if stack_int_size > 0:
                stack_int_init = loma_ir.Declare(
                    '_t_int',
                    loma_ir.Array(loma_ir.Int(), stack_int_size)
                )
                pointer_int_init = loma_ir.Declare(
                    '_stack_ptr_int',
                    loma_ir.Int(),
                    loma_ir.ConstInt(0)
                )
                new_body += [stack_int_init, pointer_int_init]
            for stmt in node.body:
                new_body.append(self.mutate_stmt(stmt))
            new_body = irmutator.flatten(new_body)
            return new_body, self.arg_id_to_diff_id
        
        def mutate_return(self, node):
            if isinstance(node.val, loma_ir.Var):
                if node.val.id == 'ret':
                    return []
            new_stmt = loma_ir.Declare('ret', self.ret_type, node.val)
            return [new_stmt]

        def mutate_declare(self, node):
            self.arg_id_to_diff_id[node.target] = '_d' + node.target + '_' + random_id_generator()
            match node.t:
                case loma_ir.Struct():
                        new_stmt = loma_ir.Declare(
                        self.arg_id_to_diff_id[node.target],
                        loma_ir.Struct(node.t.id, node.t.members)
                    )
                case _:
                    new_stmt = loma_ir.Declare(
                        self.arg_id_to_diff_id[node.target],
                        node.t
                    )
            return [node, new_stmt]

        def mutate_assign(self, node):

            if check_lhs_is_output_args(node.target, self.output_args):
                return []
            new_stmts = []
            match node.target.t:
                case loma_ir.Float():
                    new_stmts += [loma_ir.Assign(
                        loma_ir.ArrayAccess(
                            loma_ir.Var('_t_float'),
                            loma_ir.Var('_stack_ptr_float')
                        ),
                        node.target
                    )]
                    new_stmts += [loma_ir.Assign(
                        loma_ir.Var('_stack_ptr_float'),
                        loma_ir.BinaryOp(
                            loma_ir.Add(),
                            loma_ir.Var('_stack_ptr_float'),
                            loma_ir.ConstInt(1)
                        )
                    )]
                case loma_ir.Int():
                    new_stmts += [loma_ir.Assign(
                        loma_ir.ArrayAccess(
                            loma_ir.Var('_t_int'),
                            loma_ir.Var('_stack_ptr_int')
                        ),
                        node.target
                    )]
                    new_stmts += [loma_ir.Assign(
                        loma_ir.Var('_stack_ptr_int'),
                        loma_ir.BinaryOp(
                            loma_ir.Add(),
                            loma_ir.Var('_stack_ptr_int'),
                            loma_ir.ConstInt(1)
                        )
                    )]
                case loma_ir.Struct():
                    for member in node.target.t.members:
                        member_id = member.id
                        if member.t == loma_ir.Int():
                            new_stmts.append(
                                loma_ir.Assign(
                                    loma_ir.ArrayAccess(
                                    loma_ir.Var('_t_int'),
                                    loma_ir.Var('_stack_ptr_int')
                                ),
                                    loma_ir.StructAccess(
                                        node.target,
                                        member_id,
                                        None,
                                        member.t
                                    )
                                )
                            )
                            new_stmts.append(
                                    loma_ir.Assign(
                                        loma_ir.Var('_stack_ptr_int'),
                                        loma_ir.BinaryOp(
                                            loma_ir.Add(),
                                            loma_ir.Var('_stack_ptr_int'),
                                            loma_ir.ConstInt(1)
                                        )
                                    )
                            )
                        elif member.t == loma_ir.Float():
                            new_stmts.append(
                                loma_ir.Assign(
                                    loma_ir.ArrayAccess(
                                    loma_ir.Var('_t_float'),
                                    loma_ir.Var('_stack_ptr_float')
                                ),
                                    loma_ir.StructAccess(
                                        node.target,
                                        member_id,
                                        None,
                                        member.t
                                    )
                                )
                            )
                            new_stmts.append(
                                    loma_ir.Assign(
                                        loma_ir.Var('_stack_ptr_float'),
                                        loma_ir.BinaryOp(
                                            loma_ir.Add(),
                                            loma_ir.Var('_stack_ptr_float'),
                                            loma_ir.ConstInt(1)
                                        )
                                    )
                            )
            return new_stmts + [node]

        def mutate_ifelse(self, node):
            then_stmts = [self.mutate_stmt(stmt) for stmt in node.then_stmts]
            else_stmts = [self.mutate_stmt(stmt) for stmt in node.else_stmts]
            new_then_stmts = irmutator.flatten(then_stmts)
            new_else_stmts = irmutator.flatten(else_stmts)
            new_body = [loma_ir.IfElse(node.cond, new_then_stmts, new_else_stmts)]
            return new_body
        
        def mutate_call_stmt(self, node):
            def push_to_stack(arg):
                _new_stmts = []
                match arg.t:
                    case loma_ir.Float():
                        _new_stmts.append(
                            loma_ir.Assign(
                                loma_ir.ArrayAccess(
                                    loma_ir.Var('_t_float'),
                                    loma_ir.Var('_stack_ptr_float')
                                ),
                                last_arg
                            )
                        )
                        _new_stmts.append(
                            loma_ir.Assign(
                                loma_ir.Var('_stack_ptr_float'),
                                loma_ir.BinaryOp(
                                    loma_ir.Add(),
                                    loma_ir.Var('_stack_ptr_float'),
                                    loma_ir.ConstInt(1)
                                )
                            )
                        )
                    case loma_ir.Int():
                        _new_stmts.append(
                            loma_ir.Assign(
                                loma_ir.ArrayAccess(
                                    loma_ir.Var('_t_int'),
                                    loma_ir.Var('_stack_ptr_int')
                                ),
                                last_arg
                            )
                        )
                        _new_stmts.append(
                            loma_ir.Assign(
                                loma_ir.Var('_stack_ptr_int'),
                                loma_ir.BinaryOp(
                                    loma_ir.Add(),
                                    loma_ir.Var('_stack_ptr_int'),
                                    loma_ir.ConstInt(1)
                                )
                            )
                        )
                    case loma_ir.Array():
                        for i in range(arg.t.static_size):
                            _new_stmts += push_to_stack(
                                loma_ir.ArrayAccess(
                                    arg,
                                    i
                                )
                            )
                return _new_stmts

            args = node.call.args
            new_stmts = []
            for arg in args:
                if check_lhs_is_output_args(arg, self.output_args):
                    return []
            last_arg = args[-1]
            new_stmts += push_to_stack(last_arg)
            return new_stmts + [node]

        def mutate_while(self, node):
            # HW3: TODO
            new_stmts = []
            depth = check_while_depth(node)
            counter_name = f'loop_counter_{depth}'
            new_stmts.append(
                loma_ir.Declare(
                    counter_name,
                    loma_ir.Int(),
                    loma_ir.ConstInt(0)
                )
            )
            if depth > 1:
                new_stmts.append(
                    loma_ir.Declare(
                        f'loop_counter_stack_{depth-1}',
                        loma_ir.Array(loma_ir.Int(), node.max_iter)
                    )
                )
                new_stmts.append(loma_ir.Declare(
                    f'loop_counter_stack_{depth-1}_ptr',
                    loma_ir.Int(),
                    loma_ir.ConstInt(0)
                ))
            new_body = irmutator.flatten([self.mutate_stmt(stmt) for stmt in node.body])
            new_body.append(loma_ir.Assign(
                loma_ir.Var(counter_name),
                loma_ir.BinaryOp(
                    loma_ir.Add(),
                    loma_ir.Var(counter_name),
                    loma_ir.ConstInt(1)
                )
            ))
            for i in range(len(new_body) -1, -1, -1):
                if isinstance(new_body[i], loma_ir.Declare):
                    new_stmts.append(
                        loma_ir.Declare(
                            new_body[i].target,
                            new_body[i].t
                        )
                    )
                    if new_body[i].val is not None:
                        new_body[i] = loma_ir.Assign(
                            loma_ir.Var(new_body[i].target),
                            new_body[i].val
                        )
                    else:
                        if isinstance(new_body[i].t, loma_ir.Float):
                            new_body[i] = loma_ir.Assign(
                                loma_ir.Var(new_body[i].target),
                                loma_ir.ConstFloat(0.0)
                            )
                        elif isinstance(new_body[i].t, loma_ir.Int):
                            new_body[i] = loma_ir.Assign(
                                loma_ir.Var(new_body[i].target),
                                loma_ir.ConstInt(0)
                            )
                        else:
                            del new_body[i]
            
            if depth > 1:
                for i in range(len(new_body)):
                    if isinstance(new_body[i], loma_ir.While):
                        new_body.insert(i+1, loma_ir.Assign(
                            loma_ir.Var(f'loop_counter_stack_{depth-1}_ptr',),
                            loma_ir.BinaryOp(
                                loma_ir.Add(),
                                loma_ir.Var(f'loop_counter_stack_{depth-1}_ptr'),
                                loma_ir.ConstInt(1)
                            )
                        ))
                        new_body.insert(i+1, loma_ir.Assign(
                            loma_ir.ArrayAccess(
                                loma_ir.Var(f'loop_counter_stack_{depth-1}'),
                                loma_ir.Var(f'loop_counter_stack_{depth-1}_ptr')
                            ),
                            loma_ir.Var(f'loop_counter_{depth-1}')
                        ))
                        break
                    
            new_stmts.append(
                loma_ir.While(
                    node.cond,
                    node.max_iter,
                    new_body
                )
            )
            return new_stmts

        def mutate_const_float(self, node):
            # HW2: TODO
            return [node]

        def mutate_const_int(self, node):
            # HW2: TODO
            return [node]

        def mutate_var(self, node):
            return [node]

        def mutate_array_access(self, node):
            # HW2: TODO
            return [node]

        def mutate_struct_access(self, node):
            # HW2: TODO
            return [node]

        def mutate_add(self, node):
            return [node]

        def mutate_sub(self, node):
            return [node]

        def mutate_mul(self, node):
            return [node]

        def mutate_div(self, node):
            return [node]

        def mutate_call(self, node):
            return [node]


    # Apply the differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW2: TODO
            # Mutate arguments
            self.parallel = False
            node = CallNormalizeMutator().mutate_function_def(node)
            self.arg_id_to_diff_id = {}
            new_args = []
            self.output_args = [arg.id for arg in node.args if arg.i == loma_ir.Out()]
            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    self.arg_id_to_diff_id[arg.id] = '_d' + arg.id + '_' + random_id_generator()
                    new_args.append(loma_ir.Arg(
                        self.arg_id_to_diff_id[arg.id],
                        arg.t,
                        loma_ir.Out()
                    ))
                else:
                    self.arg_id_to_diff_id[arg.id] = '_d' + arg.id + '_' + random_id_generator()
                    new_args.append(
                        loma_ir.Arg(
                            self.arg_id_to_diff_id[arg.id],
                            arg.t,
                            loma_ir.In()
                        )
                    )
            if node.ret_type is not None:
                new_args.append(loma_ir.Arg(
                    '_dret',
                    node.ret_type,
                    loma_ir.In()
                ))

            forward_body, forward_arg_to_diff_id = ForwardPassMutator().mutate_body(node)
            self.arg_id_to_diff_id.update(forward_arg_to_diff_id)
            # Mutate body
            new_body = irmutator.flatten([self.mutate_stmt(stmt) for stmt in reversed(node.body)])
            new_body = forward_body + new_body
            # New function
            new_node = loma_ir.FunctionDef(
                diff_func_id,
                new_args,
                new_body,
                node.is_simd,
                None
            )
            return new_node

        def mutate_return(self, node):
            # HW2: TODO
            # return super().mutate_return(node)
            self.adj = loma_ir.Var('_dret')
            new_stmt = self.mutate_expr(node.val)
            return new_stmt

        def mutate_declare(self, node):
            # HW2: TODO
            if node.target in self.arg_id_to_diff_id and node.val is not None:
                self.adj = loma_ir.Var(self.arg_id_to_diff_id[node.target])
                expr = node.val
                return [self.mutate_expr(expr)]
            return []

        def mutate_assign(self, node):
            # HW2: TODO
            match node.target:
                case loma_ir.ArrayAccess():
                    array_name = node.target.array.id
                    temp_str_name = '_d' + array_name + '_' + random_id_generator()
                    target_ditem = loma_ir.ArrayAccess(loma_ir.Var(self.arg_id_to_diff_id[array_name]), node.target.index)
                case loma_ir.StructAccess():
                    struct_name = node.target.struct.id
                    temp_str_name = '_d' + struct_name + '_' + random_id_generator()
                    target_ditem = loma_ir.StructAccess(loma_ir.Var(self.arg_id_to_diff_id[struct_name]), node.target.member_id)
                case loma_ir.Var():
                    temp_str_name = '_d' + node.target.id + '_' + random_id_generator()
                    target_ditem = loma_ir.Var(self.arg_id_to_diff_id[node.target.id], lineno=node.target.lineno, t=node.target.t)
            new_stmts = []
            if not check_lhs_is_output_args(node.target, self.output_args):
                match node.target.t:
                    case loma_ir.Float():
                        new_stmts.append(loma_ir.Assign(
                            loma_ir.Var('_stack_ptr_float'),
                            loma_ir.BinaryOp(
                                loma_ir.Sub(),
                                loma_ir.Var('_stack_ptr_float'),
                                loma_ir.ConstInt(1)
                            )
                        ))
                        new_stmts.append(
                            loma_ir.Assign(
                                node.target,
                                loma_ir.ArrayAccess(
                                    loma_ir.Var('_t_float'),
                                    loma_ir.Var('_stack_ptr_float')
                                )
                            )
                        )
                    case loma_ir.Int():
                        new_stmts.append(loma_ir.Assign(
                            loma_ir.Var('_stack_ptr_int'),
                            loma_ir.BinaryOp(
                                loma_ir.Sub(),
                                loma_ir.Var('_stack_ptr_int'),
                                loma_ir.ConstInt(1)
                            )
                        ))
                        new_stmts.append(
                            loma_ir.Assign(
                                node.target,
                                loma_ir.ArrayAccess(
                                    loma_ir.Var('_t_int'),
                                    loma_ir.Var('_stack_ptr_int')
                                )
                            )
                        )
                    case loma_ir.Struct():
                        for mem in node.target.t.members:
                            match mem.t:
                                case loma_ir.Float():
                                    new_stmts.append(loma_ir.Assign(
                                        loma_ir.Var('_stack_ptr_float'),
                                        loma_ir.BinaryOp(
                                            loma_ir.Sub(),
                                            loma_ir.Var('_stack_ptr_float'),
                                            loma_ir.ConstInt(1)
                                        )
                                    ))
                                    new_stmts.append(
                                        loma_ir.Assign(
                                            loma_ir.StructAccess(
                                                node.target,
                                                mem.id,
                                                t=mem.t
                                            ),
                                            loma_ir.ArrayAccess(
                                                loma_ir.Var('_t_float'),
                                                loma_ir.Var('_stack_ptr_float')
                                            )
                                        )
                                    )
                                case loma_ir.Int():
                                    new_stmts.append(loma_ir.Assign(
                                        loma_ir.Var('_stack_ptr_int'),
                                        loma_ir.BinaryOp(
                                            loma_ir.Sub(),
                                            loma_ir.Var('_stack_ptr_int'),
                                            loma_ir.ConstInt(1)
                                        )
                                    ))
                                    new_stmts.append(
                                        loma_ir.Assign(
                                            loma_ir.StructAccess(
                                                node.target,
                                                mem.id,
                                                t=mem.t
                                            ),
                                            loma_ir.ArrayAccess(
                                                loma_ir.Var('_t_int'),
                                                loma_ir.Var('_stack_ptr_int')
                                            )
                                        )
                                    )

                match node.target.t:
                    case loma_ir.Struct():
                        new_stmts.append(loma_ir.Declare(
                            temp_str_name,
                            node.target.t
                        ))
                        for mem in node.target.t.members:
                            new_stmts.append(loma_ir.Assign(
                                loma_ir.StructAccess(
                                    loma_ir.Var(temp_str_name, t=node.target.t),
                                    mem.id
                                ),
                                loma_ir.StructAccess(
                                    target_ditem,
                                    mem.id
                                )
                            ))
                    case _:
                        new_stmts.append(loma_ir.Declare(
                            temp_str_name,
                            node.target.t,
                            target_ditem
                        ))
                if node.target.t == loma_ir.Float():
                    new_stmts.append(loma_ir.Assign(
                        target_ditem,
                        loma_ir.ConstFloat(0.0)
                    ))
                elif node.target.t == loma_ir.Int():
                    new_stmts.append(loma_ir.Assign(
                        target_ditem,
                        loma_ir.ConstInt(0)
                    ))
                self.adj = loma_ir.Var(temp_str_name)
            else:
                self.adj = target_ditem
            
            expr = node.val
            new_stmts.append(self.mutate_expr(expr))
            return new_stmts

        def mutate_ifelse(self, node):
            then_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.then_stmts)]
            else_stmts = [self.mutate_stmt(stmt) for stmt in reversed(node.else_stmts)]
            new_then_stmts = irmutator.flatten(then_stmts)
            new_else_stmts = irmutator.flatten(else_stmts)
            declare_stmts = []
            for i in range(len(new_then_stmts)-1, -1, -1):
                stmt = new_then_stmts[i]
                if isinstance(stmt, loma_ir.Declare):
                    declare_stmts.append(loma_ir.Declare(stmt.target, stmt.t))
                    if stmt.val is not None:
                        new_then_stmts[i] = loma_ir.Assign(loma_ir.Var(stmt.target), stmt.val)
                    else:
                        del new_then_stmts[i]
            for i in range(len(new_else_stmts)-1, -1, -1):
                stmt = new_else_stmts[i]
                if isinstance(stmt, loma_ir.Declare):
                    declare_stmts.append(loma_ir.Declare(stmt.target, stmt.t))
                    if stmt.val is not None:
                        new_else_stmts[i] = loma_ir.Assign(loma_ir.Var(stmt.target), stmt.val)
                    else:
                        del new_else_stmts[i]
            new_body = [loma_ir.IfElse(node.cond, new_then_stmts, new_else_stmts)]
            return declare_stmts + new_body

        def mutate_call_stmt(self, node):
            # HW3: TODO
            def load_stack(last_arg):
                _new_stmts = []
                match last_arg.t:
                    case loma_ir.Float():
                        _new_stmts.append(
                            loma_ir.Assign(
                                loma_ir.Var('_stack_ptr_float'),
                                loma_ir.BinaryOp(
                                    loma_ir.Sub(),
                                    loma_ir.Var('_stack_ptr_float'),
                                    loma_ir.ConstInt(1)
                                )
                            )
                        )
                        _new_stmts.append(
                            loma_ir.Assign(
                                loma_ir.Var(last_arg.id),
                                loma_ir.ArrayAccess(
                                    loma_ir.Var('_t_float'),
                                    loma_ir.Var('_stack_ptr_float')
                                )
                            )
                        )
                    case loma_ir.Int():
                        new_stmts.append(
                            loma_ir.Assign(
                                loma_ir.Var('_stack_ptr_int'),
                                loma_ir.BinaryOp(
                                    loma_ir.Sub(),
                                    loma_ir.Var('_stack_ptr_int'),
                                    loma_ir.ConstInt(1)
                                )
                            )
                        )
                        new_stmts.append(
                            loma_ir.Assign(
                                loma_ir.Var(last_arg.id),
                                loma_ir.ArrayAccess(
                                    loma_ir.Var('_t_int'),
                                    loma_ir.Var('_stack_ptr_int')
                                )
                            )
                        )
                    case loma_ir.Array():
                        for i in range(last_arg.t.static_size-1, -1, -1):
                            _new_stmts += load_stack(
                                loma_ir.ArrayAccess(
                                    last_arg,
                                    i
                                )
                            )
                return _new_stmts

            def reassign_derivative(last_arg):
                _reassign_smts = []
                match last_arg.t:
                    case loma_ir.Array():
                        for i in range(last_arg.t.static_size-1, -1, -1):
                            _reassign_smts += [
                                loma_ir.Assign(
                                    loma_ir.ArrayAccess(loma_ir.Var(self.arg_id_to_diff_id[last_arg.id]), i),
                                    loma_ir.ConstFloat(0.0)
                                )
                            ]
                    case _:
                        _reassign_smts = [loma_ir.Assign(
                            loma_ir.Var(self.arg_id_to_diff_id[last_arg.id], t=loma_ir.Float()),
                            loma_ir.ConstFloat(0.0)
                        )]
                return _reassign_smts
            self.adj = None
            last_arg = node.call.args[-1]
            if not check_lhs_is_output_args(last_arg, self.output_args):
                new_stmts = load_stack(last_arg)
                reassign_derivative_stmts = reassign_derivative(last_arg)
            else:
                new_stmts = []
                reassign_derivative_stmts = []
            return new_stmts + self.mutate_call(node.call)  + reassign_derivative_stmts

        def mutate_while(self, node):
            # HW3: TODO
            depth = check_while_depth(node)
            counter_name = f'loop_counter_{depth}'
            cond = loma_ir.BinaryOp(loma_ir.Greater(), loma_ir.Var(counter_name), loma_ir.ConstInt(0))
            body = irmutator.flatten([self.mutate_stmt(stmt) for stmt in reversed(node.body)])
            
            declare_stmts = []
            for i in range(len(body)-1, -1, -1):
                if isinstance(body[i], loma_ir.Declare):
                    declare_stmts = [loma_ir.Declare(
                        body[i].target,
                        body[i].t
                    )] + declare_stmts
                    if body[i].val is not None:
                        body[i] = loma_ir.Assign(
                            loma_ir.Var(body[i].target),
                            body[i].val
                        )
                    else:
                        if isinstance(body[i].t, loma_ir.Float):
                            body[i] = loma_ir.Assign(
                                loma_ir.Var(body[i].target),
                                loma_ir.ConstFloat(0.0)
                            )
                        elif isinstance(body[i].t, loma_ir.Int):
                            body[i] = loma_ir.Assign(
                                loma_ir.Var(body[i].target),
                                loma_ir.ConstInt(0)
                            )
                        else:
                            del body[i]

            body.append(
                loma_ir.Assign(
                    loma_ir.Var(counter_name),
                    loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.Var(counter_name),
                        loma_ir.ConstInt(1)
                    )
                )
            )

            if depth > 1:
                for i in range(len(body)):
                    if isinstance(body[i], loma_ir.While):
                        body.insert(
                            i,
                            loma_ir.Assign(
                                loma_ir.Var(f'loop_counter_{depth-1}'),
                                loma_ir.ArrayAccess(
                                    loma_ir.Var(f'loop_counter_stack_{depth-1}'),
                                    loma_ir.Var(f'loop_counter_stack_{depth-1}_ptr')
                                )
                            )
                        )
                        body.insert(
                            i,
                            loma_ir.Assign(
                                loma_ir.Var(f'loop_counter_stack_{depth-1}_ptr'),
                                loma_ir.BinaryOp(
                                    loma_ir.Sub(),
                                    loma_ir.Var(f'loop_counter_stack_{depth-1}_ptr'),
                                    loma_ir.ConstInt(1)
                                )
                            )
                        )
                        break

            return declare_stmts + [loma_ir.While(
                cond,
                node.max_iter,
                body
            )]

        def mutate_const_float(self, node):
            # HW2: TODO
            return []

        def mutate_const_int(self, node):
            # HW2: TODO
            return []

        def mutate_var(self, node):
            # HW2: TODO
            match node.t:
                case loma_ir.Struct():
                    derive_struct = loma_ir.Var(
                        self.arg_id_to_diff_id[node.id],
                        node.lineno,
                        node.t
                    )
                    return accum_deriv(derive_struct, self.adj, False)

            if node.id in self.arg_id_to_diff_id and node.t != loma_ir.Int():
                if not self.parallel:
                    new_stmt = loma_ir.Assign(\
                        loma_ir.Var(self.arg_id_to_diff_id[node.id]),
                        loma_ir.BinaryOp(
                            loma_ir.Add(),
                            loma_ir.Var(self.arg_id_to_diff_id[node.id]), self.adj
                        ))
                else:
                    new_stmt = loma_ir.CallStmt(loma_ir.Call(
                        'atomic_add',
                        [loma_ir.Var(self.arg_id_to_diff_id[node.id]),
                        self.adj]
                    ))
                return [new_stmt]
            return []

        def mutate_array_access(self, node):
            # HW2: TODO
            def transform_array_access(_node):
                match _node.array:
                    case loma_ir.Var():
                        array_name = _node.array.id
                        return loma_ir.ArrayAccess(loma_ir.Var(self.arg_id_to_diff_id[array_name]), _node.index), _node.array.t
                    case loma_ir.ArrayAccess():
                        new_array, t = transform_array_access(_node.array)
                        return loma_ir.ArrayAccess(new_array, _node.index), t
            target_deriv, t = transform_array_access(node)
            if t != loma_ir.Int():
                new_stmt = loma_ir.Assign(
                    target_deriv,
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        target_deriv, self.adj
                    )
                )
                return [new_stmt]
            return []

        def mutate_struct_access(self, node):
            # HW2: TODO
            struct_name = node.struct.id
            if node.t != loma_ir.Int():
                new_stmt = loma_ir.Assign(
                    loma_ir.StructAccess(
                        loma_ir.Var(self.arg_id_to_diff_id[struct_name]), node.member_id
                    ),
                    loma_ir.BinaryOp(
                        loma_ir.Add(),
                        loma_ir.StructAccess(
                        loma_ir.Var(self.arg_id_to_diff_id[struct_name]), node.member_id
                    ), self.adj
                    )
                )
                return [new_stmt]
            return []

        def mutate_add(self, node):
            left_node = node.left
            right_node = node.right
            left_stmt = self.mutate_expr(left_node)
            right_stmt = self.mutate_expr(right_node)
            return left_stmt + right_stmt

        def mutate_sub(self, node):
            # HW2: TODO
            left_node = node.left
            right_node = node.right
            left_stmt = self.mutate_expr(left_node)
            self.adj = loma_ir.BinaryOp(
                loma_ir.Mul(),
                self.adj, loma_ir.ConstFloat(-1.0)
            )
            right_stmt = self.mutate_expr(right_node)
            self.adj = loma_ir.BinaryOp(
                loma_ir.Mul(),
                self.adj, loma_ir.ConstFloat(-1.0)
            )
            return left_stmt + right_stmt

        def mutate_mul(self, node):
            # HW2: TODO
            left_node = node.left
            right_node = node.right
            org_adj = self.adj
            self.adj = loma_ir.BinaryOp(
                loma_ir.Mul(),
                org_adj, right_node
            )
            left_stmt = self.mutate_expr(left_node)
            self.adj = org_adj

            self.adj = loma_ir.BinaryOp(
                loma_ir.Mul(),
                org_adj, left_node
            )
            right_stmt = self.mutate_expr(right_node)
            self.adj = org_adj
            return left_stmt + right_stmt

        def mutate_div(self, node):
            # HW2: TODO
            left_node = node.left
            right_node = node.right
            org_adj = self.adj
            self.adj = loma_ir.BinaryOp(
                loma_ir.Div(),
                org_adj, right_node
            )
            left_stmt = self.mutate_expr(left_node)
            self.adj = org_adj
            self.adj = loma_ir.BinaryOp(
                loma_ir.Div(),
                    loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.ConstFloat(-1.0),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            org_adj, left_node
                        )
                    ),
                    loma_ir.Call('pow', [right_node, loma_ir.ConstFloat(2.0)])
            )

            right_stmt = self.mutate_expr(right_node)
            self.adj = org_adj
            return left_stmt + right_stmt

        def mutate_call(self, node):
            # HW2: TODO
            match node.id:
                case 'sin':
                    expr = node.args[0]
                    org_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        org_adj,
                        loma_ir.Call(
                        'cos',
                        node.args
                        )
                    )
                    new_stmt = self.mutate_expr(expr)
                    self.adj = org_adj
                    return [new_stmt]
                case 'cos':
                    expr = node.args[0]
                    org_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            org_adj,
                            loma_ir.Call(
                            'sin',
                            node.args
                            )
                        ),
                        loma_ir.ConstFloat(-1.0)
                    )
                    new_stmt = self.mutate_expr(expr)
                    self.adj = org_adj
                    return [new_stmt]
                case 'sqrt':
                    expr = node.args[0]
                    org_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        org_adj,
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.ConstFloat(2.0),
                            loma_ir.Call(
                                'sqrt',
                                node.args
                            )
                        )
                    )
                    new_stmt = self.mutate_expr(expr)
                    self.adj = org_adj
                    return [new_stmt]
                case 'exp':
                    expr = node.args[0]
                    org_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        org_adj,
                        loma_ir.Call(
                                'exp',
                                node.args
                            )
                    )
                    new_stmt = self.mutate_expr(expr)
                    self.adj = org_adj
                    return [new_stmt]
                case 'pow':
                    x = node.args[0]
                    y = node.args[1]
                    org_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        org_adj,
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            y,
                            loma_ir.Call(
                                'pow',
                                [
                                    x,
                                    loma_ir.BinaryOp(
                                        loma_ir.Sub(),
                                        y,
                                        loma_ir.ConstFloat(1.0)
                                    )
                                ]
                            )
                        )
                    )
                    x_stmt = self.mutate_expr(x)
                    self.adj = org_adj

                    org_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        org_adj,
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.Call(
                                'pow',
                                [
                                    x,
                                    y
                                ]
                            ),
                            loma_ir.Call(
                                'log',
                                [
                                    x
                                ]
                            )
                        )
                    )
                    y_stmt = self.mutate_expr(y)
                    self.adj = org_adj
                    return [x_stmt, y_stmt]
                case 'log':
                    expr = node.args[0]
                    org_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        org_adj,
                        expr
                    )
                    new_stmt = self.mutate_expr(expr)
                    self.adj = org_adj
                    return [new_stmt]
                case 'int2float':
                    return []
                case 'float2int':
                    return []
                case 'thread_id':
                    self.parallel = True
                    return []
                case _:
                    rev_id = func_to_rev[node.id]
                    if rev_id is not None:
                        new_args = []
                        for arg in reversed(node.args):
                            new_args = [arg, loma_ir.Var(self.arg_id_to_diff_id[arg.id])] + new_args
                        if self.adj is not None:
                            new_args.append(self.adj)
                        else:
                            del new_args[-2]
                        new_stmt = loma_ir.CallStmt(loma_ir.Call(rev_id, new_args))
                        return [new_stmt]
                    return []

    return RevDiffMutator().mutate_function_def(func)
