//go:build darwin

package main

import (
	_ "embed"
	"encoding/binary"
	"flag"
	"image"
	"image/draw"
	_ "image/png"
	"log"
	"math"
	"os"
	"time"

	"github.com/gogpu/gogpu/gpu/backend/native"
	"github.com/gogpu/gogpu/gpu/types"
	"github.com/gogpu/gogpu/internal/platform"
)

// -demo=1 : checkerboard texture
// -demo=2 : logo image texture
// -demo=3 : combined (checkerboard streaming + logo overlay, instanced)

//go:embed shader.wgsl
var texturedWGSL string

const (
	demoCheckerboard = 1
	demoLogoImage    = 2
	demoStress       = 3
	logoPath         = "assets/logo.png"
)

func main() {
	demo := flag.Int("demo", demoCheckerboard, "demo mode: 1=checkerboard, 2=logo image, 3=stress (instances + streaming)")
	flag.Parse()

	plat := platform.New()
	if err := plat.Init(platform.Config{
		Title:     "GoGPU - Textured Quad (Darwin)",
		Width:     800,
		Height:    600,
		Resizable: true,
	}); err != nil {
		log.Fatal(err)
	}
	defer plat.Destroy()

	backend := native.New()
	if err := backend.Init(); err != nil {
		log.Fatal(err)
	}
	defer backend.Destroy()

	instance, err := backend.CreateInstance()
	if err != nil {
		log.Fatal(err)
	}

	hinstance, hwnd := plat.GetHandle()
	surface, err := backend.CreateSurface(instance, types.SurfaceHandle{
		Instance: hinstance,
		Window:   hwnd,
	})
	if err != nil {
		log.Fatal(err)
	}

	adapter, err := backend.RequestAdapter(instance, &types.AdapterOptions{
		PowerPreference: types.PowerPreferenceHighPerformance,
	})
	if err != nil {
		log.Fatal(err)
	}

	device, err := backend.RequestDevice(adapter, nil)
	if err != nil {
		log.Fatal(err)
	}

	queue := backend.GetQueue(device)
	format := types.TextureFormatBGRA8Unorm
	surfaceConfigured := configureSurface(backend, plat, surface, device, format)

	var (
		primaryPixels   []byte
		primaryTexW     uint32
		primaryTexH     uint32
		secondaryPixels []byte
		secondaryTexW   uint32
		secondaryTexH   uint32
	)
	switch *demo {
	case demoCheckerboard:
		primaryTexW, primaryTexH = 256, 256
		primaryPixels = makeCheckerboard(primaryTexW, primaryTexH)
	case demoLogoImage:
		var err error
		primaryPixels, primaryTexW, primaryTexH, err = loadImageRGBA(logoPath)
		if err != nil {
			log.Fatalf("failed to load image %q: %v", logoPath, err)
		}
	case demoStress:
		var err error
		primaryTexW, primaryTexH = 512, 512
		primaryPixels = makeCheckerboard(primaryTexW, primaryTexH)
		secondaryPixels, secondaryTexW, secondaryTexH, err = loadImageRGBA(logoPath)
		if err != nil {
			log.Fatalf("failed to load image %q: %v", logoPath, err)
		}
	default:
		log.Fatalf("unsupported demo %d (use 1, 2, or 3)", *demo)
	}

	texture, textureView, sampler := createTextureResources(backend, device, queue, primaryTexW, primaryTexH, primaryPixels)
	defer backend.ReleaseSampler(sampler)
	defer backend.ReleaseTextureView(textureView)
	defer backend.ReleaseTexture(texture)

	var (
		secondaryTexture     types.Texture
		secondaryTextureView types.TextureView
		secondarySampler     types.Sampler
	)
	if *demo == demoStress {
		secondaryTexture, secondaryTextureView, secondarySampler = createTextureResources(backend, device, queue, secondaryTexW, secondaryTexH, secondaryPixels)
		if secondarySampler != 0 {
			defer backend.ReleaseSampler(secondarySampler)
		}
		if secondaryTextureView != 0 {
			defer backend.ReleaseTextureView(secondaryTextureView)
		}
		if secondaryTexture != 0 {
			defer backend.ReleaseTexture(secondaryTexture)
		}
	}

	bindGroupLayout, err := backend.CreateBindGroupLayout(device, &types.BindGroupLayoutDescriptor{
		Label: "texture-bind-group-layout",
		Entries: []types.BindGroupLayoutEntry{
			{
				Binding:    0,
				Visibility: types.ShaderStageFragment,
				Sampler: &types.SamplerBindingLayout{
					Type: types.SamplerBindingTypeFiltering,
				},
			},
			{
				Binding:    1,
				Visibility: types.ShaderStageFragment,
				Texture: &types.TextureBindingLayout{
					SampleType:    types.TextureSampleTypeFloat,
					ViewDimension: types.TextureViewDimension2D,
					Multisampled:  false,
				},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer backend.ReleaseBindGroupLayout(bindGroupLayout)

	pipelineLayout, err := backend.CreatePipelineLayout(device, &types.PipelineLayoutDescriptor{
		Label:            "texture-pipeline-layout",
		BindGroupLayouts: []types.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer backend.ReleasePipelineLayout(pipelineLayout)

	bindGroup, err := backend.CreateBindGroup(device, &types.BindGroupDescriptor{
		Label:  "texture-bind-group",
		Layout: bindGroupLayout,
		Entries: []types.BindGroupEntry{
			{Binding: 0, Sampler: sampler},
			{Binding: 1, TextureView: textureView},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer backend.ReleaseBindGroup(bindGroup)

	var secondaryBindGroup types.BindGroup
	if *demo == demoStress {
		secondaryBindGroup, err = backend.CreateBindGroup(device, &types.BindGroupDescriptor{
			Label:  "texture-bind-group-secondary",
			Layout: bindGroupLayout,
			Entries: []types.BindGroupEntry{
				{Binding: 0, Sampler: secondarySampler},
				{Binding: 1, TextureView: secondaryTextureView},
			},
		})
		if err != nil {
			log.Fatal(err)
		}
		defer backend.ReleaseBindGroup(secondaryBindGroup)
	}

	shader, err := backend.CreateShaderModuleWGSL(device, texturedWGSL)
	if err != nil {
		log.Fatal(err)
	}

	pipeline, err := backend.CreateRenderPipeline(device, &types.RenderPipelineDescriptor{
		Label:            "textured-quad-pipeline",
		Layout:           pipelineLayout,
		VertexShader:     shader,
		VertexEntryPoint: "vs_main",
		FragmentShader:   shader,
		FragmentEntry:    "fs_main",
		TargetFormat:     format,
		Topology:         types.PrimitiveTopologyTriangleList,
		FrontFace:        types.FrontFaceCCW,
		CullMode:         types.CullModeNone,
		VertexBuffers: []types.VertexBufferLayout{
			{
				ArrayStride: 16,
				StepMode:    types.VertexStepModeVertex,
				Attributes: []types.VertexAttribute{
					{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
					{Format: types.VertexFormatFloat32x2, Offset: 8, ShaderLocation: 1},
				},
			},
			{
				ArrayStride: 8,
				StepMode:    types.VertexStepModeInstance,
				Attributes: []types.VertexAttribute{
					{Format: types.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 2},
				},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	vertexBuffer, indexBuffer := createGeometryBuffers(backend, device, queue)
	defer backend.ReleaseBuffer(vertexBuffer)
	defer backend.ReleaseBuffer(indexBuffer)

	instanceOffsets := []float32{0, 0}
	instanceCount := uint32(1)
	if *demo == demoStress {
		instanceOffsets = buildInstanceOffsets(64, 64, 0.9)
		instanceCount = uint32(len(instanceOffsets) / 2)
	}
	instanceBuffer := createInstanceBuffer(backend, device, queue, instanceOffsets)
	defer backend.ReleaseBuffer(instanceBuffer)

	var (
		secondaryInstanceBuffer types.Buffer
		secondaryInstanceCount  uint32
	)
	if *demo == demoStress {
		secondaryInstanceBuffer = createInstanceBuffer(backend, device, queue, []float32{0, 0})
		secondaryInstanceCount = 1
		defer backend.ReleaseBuffer(secondaryInstanceBuffer)
	}

	textureLayout := &types.ImageDataLayout{
		Offset:       0,
		BytesPerRow:  primaryTexW * 4,
		RowsPerImage: primaryTexH,
	}
	textureExtent := &types.Extent3D{
		Width:              primaryTexW,
		Height:             primaryTexH,
		DepthOrArrayLayers: 1,
	}

	start := time.Now()
	for !plat.ShouldClose() {
		for {
			event := plat.PollEvents()
			if event.Type == platform.EventNone {
				break
			}
			if event.Type == platform.EventResize {
				surfaceConfigured = configureSurface(backend, plat, surface, device, format)
			}
		}

		if !surfaceConfigured {
			continue
		}

		surfaceTex, err := backend.GetCurrentTexture(surface)
		if err != nil || surfaceTex.Status != types.SurfaceStatusSuccess || surfaceTex.Texture == 0 {
			if configureSurface(backend, plat, surface, device, format) {
				surfaceConfigured = true
			}
			continue
		}

		view := backend.CreateTextureView(surfaceTex.Texture, nil)
		if view == 0 {
			backend.ReleaseTexture(surfaceTex.Texture)
			continue
		}

		encoder := backend.CreateCommandEncoder(device)
		if encoder == 0 {
			backend.ReleaseTextureView(view)
			backend.ReleaseTexture(surfaceTex.Texture)
			continue
		}

		pass := backend.BeginRenderPass(encoder, &types.RenderPassDescriptor{
			ColorAttachments: []types.ColorAttachment{
				{
					View:       view,
					LoadOp:     types.LoadOpClear,
					StoreOp:    types.StoreOpStore,
					ClearValue: types.Color{R: 0.1, G: 0.1, B: 0.12, A: 1.0},
				},
			},
		})
		if pass == 0 {
			backend.ReleaseCommandEncoder(encoder)
			backend.ReleaseTextureView(view)
			backend.ReleaseTexture(surfaceTex.Texture)
			continue
		}

		elapsed := time.Since(start).Seconds()
		angle := elapsed * 0.8
		vertices := buildRotatingQuadVertices(angle)
		backend.WriteBuffer(queue, vertexBuffer, 0, packFloat32s(vertices))
		if *demo == demoStress {
			shift := uint32(elapsed * 12)
			updateCheckerboard(primaryPixels, primaryTexW, primaryTexH, shift, shift)
			backend.WriteTexture(queue,
				&types.ImageCopyTexture{Texture: texture},
				primaryPixels,
				textureLayout,
				textureExtent,
			)
		}

		backend.SetPipeline(pass, pipeline)
		backend.SetBindGroup(pass, 0, bindGroup, nil)
		backend.SetVertexBuffer(pass, 0, vertexBuffer, 0, 0)
		backend.SetVertexBuffer(pass, 1, instanceBuffer, 0, 0)
		backend.SetIndexBuffer(pass, indexBuffer, types.IndexFormatUint16, 0, 0)
		backend.DrawIndexed(pass, 6, instanceCount, 0, 0, 0)
		if *demo == demoStress {
			backend.SetBindGroup(pass, 0, secondaryBindGroup, nil)
			backend.SetVertexBuffer(pass, 1, secondaryInstanceBuffer, 0, 0)
			backend.DrawIndexed(pass, 6, secondaryInstanceCount, 0, 0, 0)
		}
		backend.EndRenderPass(pass)
		backend.ReleaseRenderPass(pass)

		cmd := backend.FinishEncoder(encoder)
		backend.ReleaseCommandEncoder(encoder)
		if cmd != 0 {
			backend.Submit(queue, cmd)
			backend.ReleaseCommandBuffer(cmd)
		}

		backend.Present(surface)
		backend.ReleaseTextureView(view)
		backend.ReleaseTexture(surfaceTex.Texture)
	}
}

func configureSurface(backend *native.Backend, plat platform.Platform, surface types.Surface, device types.Device, format types.TextureFormat) bool {
	width, height := plat.GetSize()
	if width <= 0 || height <= 0 {
		return false
	}

	backend.ConfigureSurface(surface, device, &types.SurfaceConfig{
		Format:      format,
		Usage:       types.TextureUsageRenderAttachment,
		Width:       uint32(width),
		Height:      uint32(height),
		AlphaMode:   types.AlphaModeOpaque,
		PresentMode: types.PresentModeFifo,
	})
	return true
}

func createGeometryBuffers(backend *native.Backend, device types.Device, queue types.Queue) (types.Buffer, types.Buffer) {
	vertices := buildRotatingQuadVertices(0)
	indices := []uint16{0, 1, 2, 0, 2, 3}

	vertexBytes := packFloat32s(vertices)
	indexBytes := packUint16s(indices)

	vertexBuffer, err := backend.CreateBuffer(device, &types.BufferDescriptor{
		Label: "quad-vertex-buffer",
		Size:  uint64(len(vertexBytes)),
		Usage: types.BufferUsageVertex | types.BufferUsageCopyDst,
	})
	if err != nil {
		log.Fatal(err)
	}
	backend.WriteBuffer(queue, vertexBuffer, 0, vertexBytes)

	indexBuffer, err := backend.CreateBuffer(device, &types.BufferDescriptor{
		Label: "quad-index-buffer",
		Size:  uint64(len(indexBytes)),
		Usage: types.BufferUsageIndex | types.BufferUsageCopyDst,
	})
	if err != nil {
		log.Fatal(err)
	}
	backend.WriteBuffer(queue, indexBuffer, 0, indexBytes)

	return vertexBuffer, indexBuffer
}

func createInstanceBuffer(backend *native.Backend, device types.Device, queue types.Queue, offsets []float32) types.Buffer {
	if len(offsets) == 0 {
		log.Fatal("instance offsets are empty")
	}
	bytes := packFloat32s(offsets)
	buffer, err := backend.CreateBuffer(device, &types.BufferDescriptor{
		Label: "instance-offset-buffer",
		Size:  uint64(len(bytes)),
		Usage: types.BufferUsageVertex | types.BufferUsageCopyDst,
	})
	if err != nil {
		log.Fatal(err)
	}
	backend.WriteBuffer(queue, buffer, 0, bytes)
	return buffer
}

func createTextureResources(backend *native.Backend, device types.Device, queue types.Queue, texWidth, texHeight uint32, pixels []byte) (types.Texture, types.TextureView, types.Sampler) {
	texture, err := backend.CreateTexture(device, &types.TextureDescriptor{
		Label: "checkerboard-texture",
		Size: types.Extent3D{
			Width:              texWidth,
			Height:             texHeight,
			DepthOrArrayLayers: 1,
		},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     types.TextureDimension2D,
		Format:        types.TextureFormatRGBA8Unorm,
		Usage:         types.TextureUsageCopyDst | types.TextureUsageTextureBinding,
	})
	if err != nil {
		log.Fatal(err)
	}

	backend.WriteTexture(queue,
		&types.ImageCopyTexture{Texture: texture},
		pixels,
		&types.ImageDataLayout{
			Offset:       0,
			BytesPerRow:  texWidth * 4,
			RowsPerImage: texHeight,
		},
		&types.Extent3D{
			Width:              texWidth,
			Height:             texHeight,
			DepthOrArrayLayers: 1,
		},
	)

	view := backend.CreateTextureView(texture, nil)
	if view == 0 {
		log.Fatal("failed to create texture view")
	}

	sampler, err := backend.CreateSampler(device, &types.SamplerDescriptor{
		Label:        "texture-sampler",
		AddressModeU: types.AddressModeClampToEdge,
		AddressModeV: types.AddressModeClampToEdge,
		AddressModeW: types.AddressModeClampToEdge,
		MagFilter:    types.FilterModeLinear,
		MinFilter:    types.FilterModeLinear,
		MipmapFilter: types.MipmapFilterModeNearest,
	})
	if err != nil {
		log.Fatal(err)
	}

	return texture, view, sampler
}

func loadImageRGBA(path string) ([]byte, uint32, uint32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, 0, 0, err
	}

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)

	return rgba.Pix, uint32(bounds.Dx()), uint32(bounds.Dy()), nil
}

func buildInstanceOffsets(gridX, gridY int, scale float32) []float32 {
	if gridX <= 0 || gridY <= 0 {
		return []float32{0, 0}
	}
	if scale <= 0 {
		scale = 1
	}
	spacingX := (2 * scale) / float32(gridX)
	spacingY := (2 * scale) / float32(gridY)
	startX := -scale + spacingX*0.5
	startY := -scale + spacingY*0.5

	offsets := make([]float32, 0, gridX*gridY*2)
	for y := 0; y < gridY; y++ {
		oy := startY + float32(y)*spacingY
		for x := 0; x < gridX; x++ {
			ox := startX + float32(x)*spacingX
			offsets = append(offsets, ox, oy)
		}
	}
	return offsets
}

func buildRotatingQuadVertices(angle float64) []float32 {
	basePos := [4][2]float32{
		{-0.75, -0.75},
		{0.75, -0.75},
		{0.75, 0.75},
		{-0.75, 0.75},
	}
	baseUV := [4][2]float32{
		{0.0, 1.0},
		{1.0, 1.0},
		{1.0, 0.0},
		{0.0, 0.0},
	}
	cosA := float32(math.Cos(angle))
	sinA := float32(math.Sin(angle))

	vertices := make([]float32, 0, 16)
	for i := 0; i < 4; i++ {
		x := basePos[i][0]
		y := basePos[i][1]
		rotX := x*cosA - y*sinA
		rotY := x*sinA + y*cosA
		vertices = append(vertices, rotX, rotY, baseUV[i][0], baseUV[i][1])
	}
	return vertices
}

func makeCheckerboard(width, height uint32) []byte {
	pixels := make([]byte, int(width*height*4))
	updateCheckerboard(pixels, width, height, 0, 0)
	return pixels
}

func updateCheckerboard(pixels []byte, width, height, shiftX, shiftY uint32) {
	if width == 0 || height == 0 {
		return
	}
	const blockSize = 32
	for y := uint32(0); y < height; y++ {
		blockY := (y/blockSize + shiftY) % 2
		for x := uint32(0); x < width; x++ {
			i := int((y*width + x) * 4)
			blockX := (x/blockSize + shiftX) % 2
			if (blockX+blockY)%2 == 0 {
				pixels[i] = 255
				pixels[i+1] = 255
				pixels[i+2] = 255
				pixels[i+3] = 255
			} else {
				pixels[i] = 30
				pixels[i+1] = 144
				pixels[i+2] = 255
				pixels[i+3] = 255
			}
		}
	}
}

func packFloat32s(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func packUint16s(values []uint16) []byte {
	out := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(out[i*2:], v)
	}
	return out
}
