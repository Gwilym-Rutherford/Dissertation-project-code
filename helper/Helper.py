class Helper(object):
    
    @classmethod
    def get_mat_struct(cls, obj, path):
        if hasattr(obj, 'dtype') and obj.dtype.names:
            for name in obj.dtype.names:
                cls.get_mat_struct(obj.flatten()[0][name], path + f"['{name}']")
            else:
                shape = obj.shape if hasattr(obj, 'shape') else "scalar"
                print(f"PATH: mat{path} | SHAPE: {shape}")



    