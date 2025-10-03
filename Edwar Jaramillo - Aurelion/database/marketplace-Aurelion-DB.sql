CREATE TABLE `clientes` (
  `id_clinte` integer PRIMARY KEY,
  `nombre_cliente` varchar(200),
  `email` varchar(60),
  `ciudad` varchar(60),
  `fecha_alta` timestamp
);

CREATE TABLE `productos` (
  `id_producto` integer PRIMARY KEY,
  `nombre_producto` varchar(200),
  `categoria` varchar(60),
  `precio_unitario` decimal
);

CREATE TABLE `ventas` (
  `id_venta` integer PRIMARY KEY,
  `fecha` timestamp,
  `id_cliente` integer,
  `nombre_cliente` varchar(200),
  `email` varchar(60),
  `medio_pago` varchar(60)
);

CREATE TABLE `detalle_ventas` (
  `id_venta` integer,
  `id_producto` integer,
  `nombre_producto` varchar(200),
  `cantidad` integer,
  `precio_unitario` integer,
  `importe` bigint
);

CREATE UNIQUE INDEX `detalle_ventas_index_0` ON `detalle_ventas` (`id_venta`, `id_producto`);

ALTER TABLE `detalle_ventas` ADD FOREIGN KEY (`id_producto`) REFERENCES `productos` (`id_producto`);

ALTER TABLE `ventas` ADD FOREIGN KEY (`id_cliente`) REFERENCES `clientes` (`id_clinte`);

ALTER TABLE `detalle_ventas` ADD FOREIGN KEY (`id_venta`) REFERENCES `ventas` (`id_venta`);
