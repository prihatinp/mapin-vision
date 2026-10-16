/* Konfigurasi bawaan MAP-IN Vision.
 * Diisi sekali oleh admin, lalu siapa pun yang membuka aplikasi langsung tersambung
 * tanpa mengetik URL atau token.
 *
 * Tiap stasiun bisa dibedakan lewat link, misalnya (cocok dibuat QR code):
 *   index.html?room=line-2&station=st-05&name=Bubut%205
 *   monitor.html?room=line-2
 */
window.MV_CONFIG = {
  workerUrl: 'https://mapin-vision-rt.prihatinpurwadi2.workers.dev',
  room: 'line-1'
};
