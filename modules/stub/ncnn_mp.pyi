from typing import Any, Type, Union, overload
from typing_extensions import Self

def version() -> str:
    """Get the ncnn version string."""
    ...

class Allocator:
    """Manage memory allocation for ncnn operations."""

    def __init__(self, *, unlocked: bool = False) -> None:
        """
        Create a new memory allocator.
        Args:
            unlocked (bool, optional, default=False): Use an unlocked pool allocator if True.
        """
        ...

    def __del__(self) -> None: ...

    def fast_malloc(self, size: int) -> int:
        """
        Allocate a block of memory from the pool.
        Args:
            size (int): The number of bytes to allocate.
        Returns:
            int: An integer representing the memory address of the new block. You can pass it as a parameter in fast_free
        """
        ...

    def fast_free(self, ptr: int) -> None:
        """
        Free a previously allocated block of memory.
        Args:
            ptr (int): The integer memory address returned by fast_malloc.
        """
        ...

class Option:
    """Manage configuration options for ncnn."""

    num_threads: int
    use_local_pool_allocator: bool
    use_vulkan_compute: bool

    def __init__(self) -> None:
        """Create a new set of default options."""
        ...

    def __del__(self) -> None: ...

    def set_blob_allocator(self, allocator: Allocator) -> None:
        """
        Set the blob allocator for this option set.
        Args:
            allocator (Allocator): The allocator to use for blob data.
        """
        ...

    def set_workspace_allocator(self, allocator: Allocator) -> None:
        """
        Set the workspace allocator for this option set.
        Args:
            allocator (Allocator): The allocator to use for workspace data.
        """
        ...

class Mat:
    """Represent an ncnn matrix or multi-dimensional tensor."""

    PIXEL_RGB: int = 1
    PIXEL_BGR: int = 2
    PIXEL_GRAY: int = 3
    PIXEL_RGBA: int = 4
    PIXEL_BGRA: int = 5

    def __init__(
        self,
        w: int = 0,
        h: int = 0,
        d: int = 0,
        c: int = 0,
        *,
        data: Any | None = None,
        elemsize: int = 0,
        elempack: int = 0,
        allocator: Allocator | None = None
    ) -> None:
        """
        Create an ncnn Mat.
        This constructor can create an empty Mat, a Mat with specified dimensions, or a Mat that references external data.
        Args:
            w (int, optional, default=0): Width
            h (int, optional, default=0): Height
            d (int, optional, default=0): Depth
            c (int, optional, default=0): Channels
            data (Any | None, optional, default=None): [Keyword-only] An external data source.
            elemsize (int, optional, default=0): [Keyword-only] The size in bytes of each element in the external data.
            elempack (int, optional, default=0): [Keyword-only] The element packing format.
            allocator (Allocator | None, optional, default=None): [Keyword-only] The custom memory allocator.
        """
        ...

    def __del__(self) -> None: ...

    def fill(self, value: float) -> None:
        """Fill the Mat with a fp value."""
        ...

    def clone(self, allocator: Allocator | None = None) -> Self:
        """
        Create a **deep** copy of the Mat.
        Args:
            allocator (Allocator, optional, default=None): A custom memory allocator for the new Mat.
        """
        ...

    def reshape(self, *args: Any) -> Self:
        """
        Change the shape of the Mat without copying data.
        This method accepts 1 to 4 integer dimensions, optionally followed by an allocator.
        Args:
            *dims (int): The new dimensions (w, h, d, c).
            allocator (Allocator, optional): A custom memory allocator.
        Examples:
            mat.reshape(64)
            mat.reshape(8, 8, 8)
            mat.reshape(8, 8, my_allocator)
        """
        ...

    def flatten(self, opt: Option) -> Self:
        """
        Flatten the Mat into a 1d vector.
        Args:
            opt (Option): Configuration options.
        """
        ...

    def convert_packing(self, elempack: int, opt: Option) -> Self:
        """
        Convert the element packing format of the Mat.
        Args:
            elempack (int): The target element packing format.
            opt (Option): Configuration options.
        """
        ...

    @property
    def dims(self) -> int: ...
    @property
    def w(self) -> int: ...
    @property
    def h(self) -> int: ...
    @property
    def d(self) -> int: ...
    @property
    def c(self) -> int: ...
    @property
    def elemsize(self) -> int: ...
    @property
    def elempack(self) -> int: ...
    @property
    def cstep(self) -> int: ...
    @property
    def data(self) -> int: ...

    def get_channel_data(self, c: int) -> int:
        """
        Get the integer memory address of a specific channel.
        Args:
            c (int): The channel index.
        Returns:
            int: An integer representing the memory address of the channel data.
        """
        ...

    @classmethod
    def from_pixels(cls, pixels: Any, type: int, w: int, h: int, stride: int, allocator: Allocator | None = None) -> Self:
        """
        Create a new Mat from a pixel buffer.
        Args:
            pixels (Any): A source buffer containing pixel data.
            type (int): The pixel format type.
            w (int): The width of the pixel data.
            h (int): The height of the pixel data.
            stride (int): The stride of the pixel data in bytes.
            allocator (Allocator, optional, default=None): A custom allocator.
        """
        ...

    @classmethod
    def from_pixels_resize(cls, pixels: Any, type: int, w: int, h: int, stride: int, target_width: int, target_height: int, allocator: Allocator | None = None) -> Self:
        """
        Create a new Mat from a pixel buffer and resize it.
        Args:
            pixels (Any): A source buffer containing pixel data.
            type (int): The pixel format type.
            w (int): The width of the source pixel data.
            h (int): The height of the source pixel data.
            stride (int): The stride of the source pixel data in bytes.
            target_width (int): The target width to resize to.
            target_height (int): The target height to resize to.
            allocator (Allocator, optional, default=None): A custom allocator.
        """
        ...

    @classmethod
    def from_pixels_roi(cls, pixels: Any, type: int, w: int, h: int, stride: int, roix: int, roiy: int, roiw: int, roih: int, allocator: Allocator | None = None) -> Self:
        """
        Create a new Mat from a region of interest (ROI) in a pixel buffer.
        Args:
            pixels (Any): A source buffer containing pixel data.
            type (int): The pixel format type (e.g., ncnn.Mat.PIXEL_RGB).
            w (int): The width of the source pixel data.
            h (int): The height of the source pixel data.
            stride (int): The stride of the source pixel data in bytes.
            roix (int): The x-coordinate of the top-left corner of the ROI.
            roiy (int): The y-coordinate of the top-left corner of the ROI.
            roiw (int): The width of the ROI.
            roih (int): The height of the ROI.
            allocator (Allocator, optional, default=None): A custom allocator.
        """
        ...

    @classmethod
    def from_pixels_roi_resize(cls, pixels: Any, type: int, w: int, h: int, stride: int, roix: int, roiy: int, roiw: int, roih: int, target_width: int, target_height: int, allocator: Allocator | None = None) -> Self:
        """
        Create a new Mat from an ROI in a pixel buffer and resize it.
        Args:
            pixels (Any): A source buffer containing pixel data.
            type (int): The pixel format type.
            w (int): The width of the source pixel data.
            h (int): The height of the source pixel data.
            stride (int): The stride of the source pixel data in bytes.
            roix (int): The x-coordinate of the top-left corner of the ROI.
            roiy (int): The y-coordinate of the top-left corner of the ROI.
            roiw (int): The width of the ROI.
            roih (int): The height of the ROI.
            target_width (int): The target width to resize to.
            target_height (int): The target height to resize to.
            allocator (Allocator, optional, default=None): A custom allocator.
        """
        ...

    def to_pixels(self, pixels: Any, type: int, stride: int) -> None:
        """
        Write the Mat data to a pixel buffer.
        Args:
            pixels (Any): A writable destination buffer.
            type (int): The desired output pixel format.
            stride (int): The stride of the destination buffer in bytes.
        """
        ...

    def to_pixels_resize(self, pixels: Any, type: int, target_width: int, target_height: int, target_stride: int) -> None:
        """
        Resize the Mat and write the data to a pixel buffer.

        Args:
            pixels (Any): A writable destination buffer.
            type (int): The desired output pixel format.
            target_width (int): The target width to resize to.
            target_height (int): The target height to resize to.
            target_stride (int): The stride of the destination buffer in bytes.
        """
        ...

    def substract_mean_normalize(self, mean_vals: Any | None, norm_vals: Any | None) -> None:
        """
        Perform subtract mean and normalize operations in-place.
        Args:
            mean_vals (Any | None): A buffer containing mean values, or None.
            norm_vals (Any | None): A buffer containing normalization values, or None.
        """
        ...
    
    def from_bytes(self, data: Any) -> None:
        """
        Fill the Mat's data from a bytes-like object in-place.
        The size of the source buffer must exactly match the total size of the Mat's data.
        This method modifies the current Mat.
        Args:
            data (Any): The source buffer to copy data from (e.g., bytes, bytearray).
        """
        ...

    def to_bytes(self) -> bytes:
        """
        Convert the Mat data to a contiguous bytes object.
        Returns:
            (bytes): A bytes object containing a snapshot of the Mat's data.
        """
        ...
    
    def copy_make_border(self, top: int, bottom: int, left: int, right: int, type: int, v: float, opt: Option, *, front: int = -1, behind: int = -1) -> Self:
        """
        Create a new Mat by adding a border to this Mat.
        Args:
            top (int): The top padding size.
            bottom (int): The bottom padding size.
            left (int): The left padding size.
            right (int): The right padding size.
            type (int): The border type (e.g., ncnn.BORDER_CONSTANT).
            v (float): The value to use for constant border type.
            opt (Option): The configuration options to use.
            front (int, optional, default=-1): [Keyword-only] The front padding size for 3D mats.
            behind (int, optional, default=-1): [Keyword-only] The behind padding size for 3D mats.
        """
        ...

    def copy_cut_border(self, top: int, bottom: int, left: int, right: int, opt: Option, *, front: int = -1, behind: int = -1) -> Self:
        """
        Create a new Mat by cutting the border from this Mat.
        Args:
            top (int): The top border size to cut.
            bottom (int): The bottom border size to cut.
            left (int): The left border size to cut.
            right (int): The right border size to cut.
            opt (Option): The configuration options to use.
            front (int, optional, default=-1): [Keyword-only] The front border size to cut for 3D mats.
            behind (int, optional, default=-1): [Keyword-only] The behind border size to cut for 3D mats.
        """
        ...

    def draw_rectangle(self, rect: tuple[int, int, int, int], color: int, thickness: int) -> None:
        """
        Draw a rectangle on the Mat in-place.

        Args:
            rect (tuple[int, int, int, int]): The rectangle to draw, formatted as (x, y, width, height).
            color (int): The color to use for drawing.
            thickness (int): The thickness of the rectangle's border.
        """
        ...

    def draw_text(self, text: str, origin: tuple[int, int], font_size: int, color: int) -> None:
        """
        Draw text on the Mat in-place.

        Args:
            text (str): The text string to draw.
            origin (tuple[int, int]): The top-left corner of the text, formatted as (x, y).
            font_size (int): The size of the font in pixels.
            color (int): The color to use for the text.
        """
        ...

    def draw_circle(self, center: tuple[int, int], radius: int, color: int, thickness: int) -> None:
        """
        Draw a circle on the Mat in-place.

        Args:
            center (tuple[int, int]): The center of the circle, formatted as (x, y).
            radius (int): The radius of the circle.
            color (int): The color to use for drawing.
            thickness (int): The thickness of the circle's border.
        """
        ...

    def draw_line(self, pt1: tuple[int, int], pt2: tuple[int, int], color: int, thickness: int) -> None:
        """
        Draw a line on the Mat in-place.

        Args:
            pt1 (tuple[int, int]): The starting point of the line, formatted as (x, y).
            pt2 (tuple[int, int]): The ending point of the line, formatted as (x, y).
            color (int): The color to use for drawing.
            thickness (int): The thickness of the line.
        """
        ...

class Blob:
    """
    Represent a data blob in the ncnn computation graph.
    Note: Blob objects are typically not created by users, but are returned
    by other parts of the library, such as from a network layer.
    """

    @property
    def name(self) -> str: ...

    @property
    def producer(self) -> int: ...

    @property
    def consumer(self) -> int: ...

    @property
    def shape(self) -> tuple[int, int, int, int]:  ...

class ParamDict:
    """Manages parameters for a layer, accessed by an integer ID."""

    def __init__(self) -> None: ...

    def __del__(self) -> None: ...

    def get_type(self, id: int) -> int:
        """
        Get the data type of a parameter.
        Args:
            id (int): The parameter ID.
        """
        ...

    def get_int(self, id: int, default: int) -> int:
        """
        Get a parameter as an integer.
        Args:
            id (int): The parameter ID.
            default (int): The default value to return if the ID is not found.
        """
        ...

    def get_float(self, id: int, default: float) -> float:
        """
        Get a parameter as a float.
        Args:
            id (int): The parameter ID.
            default (float): The default value to return if the ID is not found.
        """
        ...

    def get_array(self, id: int, default: Mat) -> Mat | None:
        """
        Get a parameter as a Mat object.
        Args:
            id (int): The parameter ID.
            default (Mat): The default Mat to return if the ID is not found.
        """
        ...

    def set_int(self, id: int, i: int) -> None:
        """
        Set an integer parameter.
        Args:
            id (int): The parameter ID.
            i (int): The integer value to set.
        """
        ...

    def set_float(self, id: int, f: float) -> None:
        """
        Set a float parameter.
        Args:
            id (int): The parameter ID.
            f (float): The float value to set.
        """
        ...

    def set_array(self, id: int, v: Mat) -> None:
        """
        Set a Mat parameter.
        Args:
            id (int): The parameter ID.
            v (Mat): The Mat object to set.
        """
        ...

class DataReader:
    """Read binary data from various sources."""

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, *, from_memory: Any) -> None: ...
    @overload
    def __init__(self, *, from_stdio: int) -> None: ...

    def __del__(self) -> None: ...

    def scan(self, format: str, data: Any) -> int:
        """
        Read structured binary data from the stream.
        Args:
            format (str): A format string specifying the data structure.
            data (Any): A writable buffer to store the scanned data.
        Returns:
            int: The number of items successfully scanned.
        """
        ...

    def read(self, buffer: Any) -> int:
        """
        Read raw bytes from the stream into a buffer.
        Args:
            buffer (Any): A writable buffer to fill with data.
        Returns:
            int: The number of bytes actually read.
        """
        ...

class ModelBin:
    """Represent the binary weights of a model."""

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, *, from_datareader: DataReader) -> None: ...
    @overload
    def __init__(self, *, from_mat_array: list[Mat]) -> None: ...

    def __del__(self) -> None: ...

    def load(self, *args: int) -> Mat | None:
        """
        Load a weight Mat from the model binary.
        Args:
            *dims (int): The dimensions of the Mat to load (e.g., w, h, c).
            type (int): The data type of the Mat to load.
        Returns:
            (Mat | None): The loaded Mat object, or None if loading fails.
        """
        ...

class Layer:
    """Represent a single layer in an ncnn neural network."""

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, *, type: str) -> None: ...
    @overload
    def __init__(self, *, typeindex: int) -> None: ...

    def __del__(self) -> None: ...

    one_blob_only: bool
    support_inplace: bool
    support_vulkan: bool
    support_packing: bool
    support_bf16_storage: bool
    support_fp16_storage: bool

    @property
    def name(self) -> str: ...

    @property
    def typeindex(self) -> int: ...

    @property
    def type(self) -> str: ...
    
    @property
    def bottom_count(self) -> int: ...

    @property
    def top_count(self) -> int: ...
    
    @staticmethod
    def type_to_index(type: str) -> int:
        """
        Convert a layer type name to its corresponding type index.
        Args:
            type (str): The name of the layer type.
        """
        ...
    
    def get_bottom(self, i: int) -> int:
        """
        Get the index of a bottom (input) blob.
        Args:
            i (int): The index of the bottom blob to retrieve.
        """
        ...

    def get_top(self, i: int) -> int:
        """
        Get the index of a top (output) blob.
        Args:
            i (int): The index of the top blob to retrieve.
        """
        ...
    
    def get_bottom_shape(self, i: int) -> tuple[int, int, int, int]:
        """
        Get the shape of a bottom (input) blob.
        Args:
            i (int): The index of the bottom blob.
        Returns:
            (tuple[int, int, int, int]): The shape of the blob as (dims, w, h, c).
        """
        ...

    def get_top_shape(self, i: int) -> tuple[int, int, int, int]:
        """
        Get the shape of a top (output) blob.
        Args:
            i (int): The index of the top blob.
        Returns:
            (tuple[int, int, int, int]): The shape of the blob as (dims, w, h, c).
        """
        ...

    def load_param(self, pd: ParamDict) -> int:
        """
        Load layer parameters from a ParamDict.
        Args:
            pd (ParamDict): The parameter dictionary to load from.
        Returns:
            (int): An integer status code, typically 0 for success.
        """
        ...

    def load_model(self, mb: ModelBin) -> int:
        """
        Load layer weights from a ModelBin.
        Args:
            mb (ModelBin): The model binary to load weights from.
        Returns:
            (int): An integer status code, typically 0 for success.
        """
        ...

    def create_pipeline(self, opt: Option) -> int:
        """
        Create the layer's internal pipeline for computation.
        Args:
            opt (Option): The configuration options to use.
        Returns:
            (int): An integer status code, typically 0 for success.
        """
        ...

    def destroy_pipeline(self, opt: Option) -> int:
        """
        Destroy the layer's internal pipeline.
        Args:
            opt (Option): The configuration options used during creation.
        Returns:
            (int): An integer status code, typically 0 for success.
        """
        ...

    @overload
    def forward(self, bottom_blob: Mat, opt: Option) -> Mat: ...
    @overload
    def forward(self, bottom_blobs: list[Mat], top_blobs_count: int, opt: Option) -> tuple[Mat, ...]: ...
    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the layer.
        This method has two forms depending on the input:
        1. Single-input, single-output.
        2. Multi-input, multi-output.
        Args:
            bottom_blob (Mat): The single input Mat.
            bottom_blobs (list[Mat]): A list of input Mats.
            top_blobs_count (int): The expected number of output Mats.
            opt (Option): The configuration options to use for the forward pass.
        Returns:
            (Mat | tuple[Mat, ...]): A single output Mat or a tuple of output Mats.
        """
        ...

    @overload
    def forward_inplace(self, bottom_top_blob: Mat, opt: Option) -> int: ...
    @overload
    def forward_inplace(self, bottom_top_blobs: list[Mat], opt: Option) -> int: ...
    def forward_inplace(self, *args, **kwargs):
        """
        Perform an in-place forward pass through the layer.
        The input blob(s) are modified directly to store the output.
        Args:
            bottom_top_blob (Mat): The single Mat to process in-place.
            bottom_top_blobs (list[Mat]): The list of Mats to process in-place.
            opt (Option): The configuration options to use.
        Returns:
            (int): An integer status code, typically 0 for success.
        """
        ...

class Net:
    """Represents a neural network for inference."""

    option: Option

    @property
    def input_count(self) -> int: ...
    @property
    def output_count(self) -> int: ...

    def __init__(self) -> None: ...

    def __del__(self) -> None: ...

    def set_vulkan_device(self, device_index: int) -> None:
        """
        Set the Vulkan device to use for inference.
        Note: This method is only available if compiled with NCNN_VULKAN=ON.
        Args:
            device_index (int): The index of the Vulkan device.
        """
        ...

    @overload
    def register_custom_layer(self, type: str, layer_class: Type[Layer]) -> None:
        """
        Register a custom layer class with the network using its type name.

        Args:
            type (str): The type name of the custom layer (e.g., "MyLayer").
            layer_class (Type[Layer]): The Python class that implements the custom layer logic.
        """
        ...
    @overload
    def register_custom_layer(self, typeindex: int, layer_class: Type[Layer]) -> None:
        """
        Register a custom layer class with the network using its type index.

        Args:
            typeindex (int): The type index of the custom layer.
            layer_class (Type[Layer]): The Python class that implements the custom layer logic.
        """
        ...
    def register_custom_layer(self, identifier: Union[str, int], layer_class: Type[Layer]) -> None: ...

    def load_param(self, source: str | DataReader) -> None:
        """
        Load the network structure from a .param file or a DataReader.
        Args:
            source (str | DataReader): The file path to the .param file or a DataReader instance.
        """
        ...

    def load_param_bin(self, source: str | DataReader | Any) -> None:
        """
        Load the network structure from a compiled .param.bin file or other sources.
        Args:
            source (str | DataReader | Any): The file path, a DataReader, or a buffer-like object.
        """
        ...

    def load_model(self, source: str | DataReader | Any) -> None:
        """
        Load the network weights from a .bin file or other sources.
        Args:
            source (str | DataReader | Any): The file path, a DataReader, or a buffer-like object.
        """
        ...

    def clear(self) -> None:
        """Clear the network, releasing its internal structures and weights."""
        ...

    def get_input_name(self, i: int) -> str:
        """
        Get the name of an input blob by its index.
        Note: This method is only available if compiled with NCNN_STRING=ON.
        Args:
            i (int): The index of the input blob.
        """
        ...

    def get_output_name(self, i: int) -> str:
        """
        Get the name of an output blob by its index.
        Note: This method is only available if compiled with NCNN_STRING=ON.
        Args:
            i (int): The index of the output blob.
        """
        ...

    def get_input_index(self, i: int) -> int:
        """
        Get the internal blob index for a given input index.
        Args:
            i (int): The index of the input blob.
        """
        ...

    def get_output_index(self, i: int) -> int:
        """
        Get the internal blob index for a given output index.
        Args:
            i (int): The index of the output blob.
        """
        ...

    def create_extractor(self) -> 'Extractor':
        """Create a new Extractor to run the network and extract features."""
        ...

class Extractor:
    """
    Extract features from the intermediate layers of a neural network.
    """

    def __del__(self) -> None: ...

    def set_option(self, opt: Option) -> None:
        """
        Set the options for this extractor.
        Args:
            opt (Option): The configuration options to use.
        """
        ...

    def input(self, id: str | int, in_mat: Mat) -> int:
        """
        Input a Mat into the network at a specific blob.
        Args:
            id (str | int): The name or index of the input blob.
            in_mat (Mat): The input Mat object.
        Returns:
            (int): An integer status code, typically 0 for success.
        """
        ...

    def extract(self, id: str | int) -> Mat:
        """
        Extract a Mat from the network at a specific blob.
        Args:
            id (str | int): The name or index of the output blob to extract.
        Returns:
            (Mat): The extracted Mat object.
        """
        ...
