"""
Visualização de Cobertura Esférica da Imagem no Dataset Lunar

Este script pega uma imagem de exemplo do dataset, extrai suas coordenadas
geográficas (latitude e longitude das bordas) e altitude, e cria uma 
visualização 3D de uma esfera representando a Lua com a região coberta
pela imagem pintada em verde.
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import OpenEXR
import Imath

# Constantes
MOON_RADIUS_KM = 1737.4

def read_exr_channels(exr_path):
    """
    Lê os canais R e G de um arquivo EXR (latitude e longitude).
    """
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = exr_file.channel('R', FLOAT)
    g_str = exr_file.channel('G', FLOAT)
    
    lat_map = np.frombuffer(r_str, dtype=np.float32).reshape(size[1], size[0])
    lon_map = np.frombuffer(g_str, dtype=np.float32).reshape(size[1], size[0])
    
    return lat_map, lon_map

def latlon_to_cartesian(lat, lon, alt):
    """
    Converte coordenadas geográficas para cartesianas 3D.
    """
    r = MOON_RADIUS_KM + alt
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    
    return x, y, z

def create_sphere(radius, resolution=50):
    """
    Cria uma malha esférica para visualização.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    return x, y, z

def get_image_bounds(lat_map, lon_map):
    """
    Extrai os limites geográficos da imagem (bordas).
    """
    bounds = {
        'lat_min': lat_map.min(),
        'lat_max': lat_map.max(),
        'lon_min': lon_map.min(),
        'lon_max': lon_map.max(),
        'lat_center': lat_map.mean(),
        'lon_center': lon_map.mean()
    }
    return bounds

def visualize_all_coverage(dataset_root):
    """
    Cria visualizações 3D mostrando todas as regiões do dataset cobertas na Lua.
    Gera múltiplas visões com diferentes ângulos.
    """
    # 1. Carregar arquivos
    exr_files = sorted(glob.glob(os.path.join(dataset_root, "lat_lon_exr", "*.exr")))
    alt_files = sorted(glob.glob(os.path.join(dataset_root, "altimeter", "*.txt")))
    
    if not exr_files or not alt_files:
        print("❌ Erro: Dataset não encontrado ou incompleto!")
        return
    
    # Usar apenas arquivos correspondentes
    exr_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in exr_files}
    alt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in alt_files}
    
    common_names = sorted(set(exr_dict.keys()) & set(alt_dict.keys()))
    
    print("=" * 60)
    print(f"📊 VISUALIZANDO COBERTURA COMPLETA DO DATASET")
    print(f"   Total de amostras: {len(common_names)}")
    print("=" * 60)
    
    # Definir múltiplas visões (elevação, azimute)
    views = [
        (20, 0, "Visão Frontal"),
        (20, 90, "Visão Lateral Direita"),
        (20, 180, "Visão Traseira"),
        (20, 270, "Visão Lateral Esquerda"),
        (90, 0, "Visão Polo Norte"),
        (-90, 0, "Visão Polo Sul")
    ]
    
    for view_idx, (elev, azim, view_name) in enumerate(views):
        print(f"\n🎨 Gerando {view_name}...")
        
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111, projection='3d')
        
        # Criar esfera base (Lua) em cinza claro
        x_sphere, y_sphere, z_sphere = create_sphere(MOON_RADIUS_KM, resolution=100)
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.2, 
                       edgecolor='none', linewidth=0)
        
        # Processar todas as amostras
        for idx, sample_name in enumerate(common_names):
            if idx % 1000 == 0:
                print(f"   Processando amostra {idx}/{len(common_names)}...")
            
            exr_path = exr_dict[sample_name]
            alt_path = alt_dict[sample_name]
            
            # Carregar dados
            lat_map, lon_map = read_exr_channels(exr_path)
            
            with open(alt_path, 'r') as f:
                altitude = float(f.read().strip())
            
            # Subsampling para não sobrecarregar a visualização
            step = max(1, lat_map.shape[0] // 20)
            lat_sub = lat_map[::step, ::step]
            lon_sub = lon_map[::step, ::step]
            
            # Converter pontos da imagem para coordenadas cartesianas
            x_img, y_img, z_img = latlon_to_cartesian(lat_sub, lon_sub, altitude)
            
            # Plotar a região coberta em verde
            ax.plot_surface(x_img, y_img, z_img, color='green', alpha=0.7, 
                           edgecolor='none', linewidth=0)
        
        # Configurar visualização
        ax.set_xlabel('X (km)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (km)', fontsize=14, fontweight='bold')
        ax.set_zlabel('Z (km)', fontsize=14, fontweight='bold')
        ax.set_title(f'Cobertura Completa do Dataset - {view_name}\n{len(common_names)} regiões', 
                    fontsize=16, fontweight='bold')
        
        # Ajustar limites dos eixos
        max_range = MOON_RADIUS_KM * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # Aspecto igual
        ax.set_box_aspect([1, 1, 1])
        
        # Ajustar ângulo de visualização
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        
        # Salvar figura
        output_path = f"coverage_all_view_{view_idx+1}_{view_name.replace(' ', '_').lower()}.png"
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"   ✅ Salvo: {output_path}")
        
        plt.close(fig)
    
    print("\n" + "=" * 60)
    print(f"✅ Todas as visualizações foram geradas com sucesso!")
    print("=" * 60)

def visualize_moon_coverage(dataset_root, sample_idx=0):
    """
    Cria uma visualização 3D mostrando a região da Lua coberta por uma imagem.
    """
    # 1. Carregar arquivos
    img_files = sorted(glob.glob(os.path.join(dataset_root, "img", "*.png")))
    exr_files = sorted(glob.glob(os.path.join(dataset_root, "lat_lon_exr", "*.exr")))
    alt_files = sorted(glob.glob(os.path.join(dataset_root, "altimeter", "*.txt")))
    
    if not img_files or not exr_files or not alt_files:
        print("❌ Erro: Dataset não encontrado ou incompleto!")
        return
    
    # Usar apenas arquivos correspondentes
    img_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
    exr_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in exr_files}
    alt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in alt_files}
    
    common_names = sorted(set(img_dict.keys()) & set(exr_dict.keys()) & set(alt_dict.keys()))
    
    if sample_idx >= len(common_names):
        print(f"❌ Erro: Índice {sample_idx} fora do range. Existem {len(common_names)} amostras.")
        return
    
    sample_name = common_names[sample_idx]
    img_path = img_dict[sample_name]
    exr_path = exr_dict[sample_name]
    alt_path = alt_dict[sample_name]
    
    # 2. Carregar dados
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    lat_map, lon_map = read_exr_channels(exr_path)
    
    with open(alt_path, 'r') as f:
        altitude = float(f.read().strip())
    
    # 3. Extrair informações das bordas
    bounds = get_image_bounds(lat_map, lon_map)
    
    print("=" * 60)
    print(f"📷 Amostra: {sample_name}")
    print(f"   Resolução da Imagem: {image.shape}")
    print("=" * 60)
    print("\n📍 COORDENADAS GEOGRÁFICAS:")
    print(f"   Latitude:")
    print(f"      Min: {bounds['lat_min']:.4f}°")
    print(f"      Max: {bounds['lat_max']:.4f}°")
    print(f"      Centro: {bounds['lat_center']:.4f}°")
    print(f"   Longitude:")
    print(f"      Min: {bounds['lon_min']:.4f}°")
    print(f"      Max: {bounds['lon_max']:.4f}°")
    print(f"      Centro: {bounds['lon_center']:.4f}°")
    print(f"\n🏔️  Altitude: {altitude:.2f} km")
    print("=" * 60)
    
    # 4. Criar visualização 3D
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Imagem Original
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Imagem Original\n{sample_name}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Subplot 2: Mapa de Latitude
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(lat_map, cmap='viridis')
    ax2.set_title('Mapa de Latitude (°)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    
    # Subplot 3: Visualização 3D da Esfera
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Criar esfera base (Lua) em cinza claro
    sphere_radius = MOON_RADIUS_KM + altitude
    x_sphere, y_sphere, z_sphere = create_sphere(MOON_RADIUS_KM, resolution=50)
    ax3.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.3, 
                     edgecolor='none', linewidth=0)
    
    # Criar malha da região coberta pela imagem
    # Subsampling para não sobrecarregar a visualização
    step = max(1, lat_map.shape[0] // 50)  # Aproximadamente 50 pontos por dimensão
    lat_sub = lat_map[::step, ::step]
    lon_sub = lon_map[::step, ::step]
    
    # Converter pontos da imagem para coordenadas cartesianas
    x_img, y_img, z_img = latlon_to_cartesian(lat_sub, lon_sub, altitude)
    
    # Plotar a região coberta em verde
    ax3.plot_surface(x_img, y_img, z_img, color='green', alpha=0.8, 
                     edgecolor='darkgreen', linewidth=0.5)
    
    # Configurar visualização
    ax3.set_xlabel('X (km)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Y (km)', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Z (km)', fontsize=10, fontweight='bold')
    ax3.set_title('Cobertura na Esfera Lunar\n(Verde = Região da Imagem)', 
                  fontsize=12, fontweight='bold')
    
    # Ajustar limites dos eixos para mostrar a esfera completa
    max_range = sphere_radius * 1.1
    ax3.set_xlim([-max_range, max_range])
    ax3.set_ylim([-max_range, max_range])
    ax3.set_zlim([-max_range, max_range])
    
    # Aspecto igual para manter a esfera circular
    ax3.set_box_aspect([1, 1, 1])
    
    # Ajustar ângulo de visualização para melhor perspectiva
    ax3.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Salvar figura
    output_path = f"visualization_{sample_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualização salva em: {output_path}")
    
    plt.show()

def main():
    """
    Função principal - executa a visualização.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualizar cobertura esférica de imagem lunar')
    parser.add_argument('--dataset', type=str, default='../LINNA-Crater/LunarLanding_Dataset/LunarLanding_Dataset',
                        help='Caminho para o dataset')
    parser.add_argument('--sample', type=int, default=10000,
                        help='Índice da amostra a visualizar (padrão: 10000)')
    parser.add_argument('--all', action='store_true',
                        help='Visualizar todas as regiões do dataset com múltiplas visões')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ Erro: Dataset não encontrado em {args.dataset}")
        print("   Ajuste o caminho usando --dataset <caminho>")
        return
    
    if args.all:
        visualize_all_coverage(args.dataset)
    else:
        visualize_moon_coverage(args.dataset, args.sample)

if __name__ == "__main__":
    main()
