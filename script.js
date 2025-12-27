// Este es tu inventario de repuestos
const productos = [
    { 
        id: 1, 
        nombre: "Compresor Denso 10P15C", 
        marca: "toyota", 
        modelo: "corolla", 
        precio: 250, 
        img: "https://via.placeholder.com/150" 
    },
    { 
        id: 2, 
        nombre: "Evaporador de Aluminio", 
        marca: "toyota", 
        modelo: "hilux", 
        precio: 110, 
        img: "https://via.placeholder.com/150" 
    },
    { 
        id: 3, 
        nombre: "Condensador de Flujo Paralelo", 
        marca: "ford", 
        modelo: "f-150", 
        precio: 185, 
        img: "https://via.placeholder.com/150" 
    },
    { 
        id: 4, 
        nombre: "Filtro de Cabina Carbón", 
        marca: "chevrolet", 
        modelo: "aveo", 
        precio: 20, 
        img: "https://via.placeholder.com/150" 
    },
    { 
        id: 5, 
        nombre: "Válvula de Expansión", 
        marca: "toyota", 
        modelo: "yaris", 
        precio: 45, 
        img: "https://via.placeholder.com/150" 
    }
];const contenedorProductos = document.getElementById('contenedor-productos');

function renderizarProductos(listaParaMostrar) {
    // 1. Limpiamos el contenedor para que no se dupliquen
    contenedorProductos.innerHTML = "";

    // 2. Si la lista está vacía, mostramos un mensaje
    if (listaParaMostrar.length === 0) {
        contenedorProductos.innerHTML = `<p style="text-align:center; width:100%;">No hay repuestos disponibles para este criterio.</p>`;
        return;
    }

    // 3. Creamos las tarjetas para cada producto de la lista
    listaParaMostrar.forEach(producto => {
        const tarjeta = document.createElement('div');
        tarjeta.classList.add('product-card');

        tarjeta.innerHTML = `
            <img src="${producto.img}" alt="${producto.nombre}">
            <div class="card-body">
                <span class="category">${producto.marca.toUpperCase()}</span>
                <h4>${producto.nombre}</h4>
                <p class="compatibility">Modelo: ${producto.modelo}</p>
                <p class="price">$${producto.precio}</p>
                <button class="add-btn" onclick="agregarAlCarrito()">Añadir al carrito</button>
            </div>
        `;
        contenedorProductos.appendChild(tarjeta);
    });
}

// Llamamos a la función al cargar la página para ver todos los productos inicialmente
renderizarProductos(productos);const btnFiltrar = document.getElementById('btn-buscar');

btnFiltrar.addEventListener('click', () => {
    const marcaSeleccionada = document.getElementById('select-marca').value;
    const modeloSeleccionado = document.getElementById('select-modelo').value;

    // Filtramos el array principal
    const productosFiltrados = productos.filter(item => {
        // Si no seleccionó marca, muestra todo. Si seleccionó, compara.
        const coincideMarca = marcaSeleccionada === "" || item.marca === marcaSeleccionada;
        // Si no seleccionó modelo, muestra todo de esa marca. Si seleccionó, compara.
        const coincideModelo = modeloSeleccionado === "" || item.modelo === modeloSeleccionado;
        
        return coincideMarca && coincideModelo;
    });

    // Actualizamos la pantalla con los resultados del filtro
    renderizarProductos(productosFiltrados);
});


// Variables iniciales
let contadorCarrito = 0;
const botonesAgregar = document.querySelectorAll('.add-btn');
const visualizadorContador = document.getElementById('cart-count');

// Función para añadir al carrito
botonesAgregar.forEach((boton) => {
    boton.addEventListener('click', () => {
        // 1. Aumentar el contador
        contadorCarrito++;
        
        // 2. Actualizar el número en la pantalla
        visualizadorContador.innerText = contadorCarrito;
        
        // 3. Efecto visual simple
        boton.innerText = "✅ ¡Añadido!";
        boton.style.backgroundColor = "#28a745"; // Cambia a verde

        // Regresar el botón a su estado original después de 1.5 segundos
        setTimeout(() => {
            boton.innerText = "Añadir al carrito";
            boton.style.backgroundColor = "#003366";
        }, 1500);

        // Mensaje en consola (para pruebas)
        console.log("Producto añadido. Total en carrito: " + contadorCarrito);
    });
});
// Base de datos de modelos
const modelosPorMarca = {
    toyota: ["Corolla", "Hilux", "Yaris", "Terios"],
    ford: ["F-150", "Fiesta", "Explorer", "EcoSport"],
    chevrolet: ["Aveo", "Silverado", "Spark", "Cruze"]
};

const selectMarca = document.getElementById('select-marca');
const selectModelo = document.getElementById('select-modelo');

// Escuchar cuando el usuario cambia la marca
selectMarca.addEventListener('change', function() {
    const marcaSeleccionada = this.value;

    // Limpiar los modelos anteriores
    selectModelo.innerHTML = '<option value="">Selecciona Modelo</option>';

    if (marcaSeleccionada !== "") {
        // Obtener los modelos de esa marca
        const modelos = modelosPorMarca[marcaSeleccionada];

        // Crear una opción por cada modelo
        modelos.forEach(function(modelo) {
            const opcion = document.createElement('option');
            opcion.value = modelo.toLowerCase();
            opcion.text = modelo;
            selectModelo.appendChild(opcion);
        });
    }
});
let carrito = []; // Lista de productos seleccionados

// Función para abrir/cerrar modal
const modal = document.getElementById("modal-carrito");
const btnAbrirCarrito = document.querySelector(".cart-btn");
const btnCerrarCarrito = document.querySelector(".close-btn");

btnAbrirCarrito.onclick = () => {
    actualizarVistaCarrito();
    modal.style.display = "block";
};

btnCerrarCarrito.onclick = () => modal.style.display = "none";

// Función para agregar productos al carrito
function agregarAlCarrito(id) {
    const productoEncontrado = productos.find(p => p.id === id);
    carrito.push(productoEncontrado);
    
    // Actualizar contador visual
    document.getElementById('cart-count').innerText = carrito.length;
    alert(`${productoEncontrado.nombre} añadido al carrito`);
}

// Función para mostrar los productos dentro del modal
function actualizarVistaCarrito() {
    const contenedorItems = document.getElementById('lista-carrito-items');
    const totalPrecio = document.getElementById('total-precio');
    
    contenedorItems.innerHTML = "";
    let total = 0;

    carrito.forEach((item, index) => {
        total += item.price || item.precio; // Asegúrate que el campo se llame igual
        const div = document.createElement('div');
        div.style.display = "flex";
        div.style.justifyContent = "space-between";
        div.style.marginBottom = "10px";
        div.innerHTML = `
            <span>${item.nombre}</span>
            <strong>$${item.precio}</strong>
        `;
        contenedorItems.appendChild(div);
    });

    totalPrecio.innerText = total;
}// Función para enviar el pedido por WhatsApp
function enviarWhatsApp() {
    if (carrito.length === 0) {
        alert("Tu carrito está vacío.");
        return;
    }

    // 1. Configura tu número de teléfono (con código de país, sin el signo +)
    // Ejemplo: 584120000000 para Venezuela o 521... para México
    const miNumero = "TUNUMEROAQUI"; 

    // 2. Construir el texto del mensaje
    const marcaSeleccionada = document.getElementById('select-marca').value;
const modeloSeleccionado = document.getElementById('select-modelo').value;

mensaje += `%0A%0A*Vehículo:* ${marcaSeleccionada.toUpperCase()} ${modeloSeleccionado.toUpperCase()}`;

    carrito.forEach((prod, index) => {
        mensaje += `*${index + 1}.* ${prod.nombre} - $${prod.precio}%0A`;
        total += prod.precio;
    });

    mensaje += `%0A*Total a pagar: $${total}*`;
    mensaje += "%0A%0A¿Podrían confirmarme disponibilidad?";

    // 3. Crear el link de WhatsApp
    const url = `https://wa.me/${miNumero}?text=${mensaje}`;

    // 4. Abrir en una nueva pestaña
    window.open(url, '_blank');
}

// Conectar la función al botón del modal
document.querySelector(".checkout-btn").onclick = enviarWhatsApp;
