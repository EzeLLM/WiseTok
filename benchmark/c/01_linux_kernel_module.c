/*
 * example_device.c - A simple Linux kernel module demonstrating device driver patterns
 *
 * This module implements a character device with file operations, ioctl handling,
 * and kernel memory copy semantics typical of Linux kernel code.
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/ioctl.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/types.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Device Driver Example");
MODULE_DESCRIPTION("A simple character device driver");
MODULE_VERSION("0.1");

#define DEVICE_NAME "example_dev"
#define CLASS_NAME "example_class"
#define IOCTL_SET_VALUE _IOW('E', 1, unsigned int)
#define IOCTL_GET_VALUE _IOR('E', 2, unsigned int)
#define BUFFER_SIZE 256

static int device_major = 0;
static struct class *device_class = NULL;
static struct device *device = NULL;
static struct cdev device_cdev;
static unsigned int device_value = 42;
static char device_buffer[BUFFER_SIZE];

/**
 * example_dev_read - Read from the device
 * @file: File structure pointer
 * @user_buffer: User-space buffer to write to
 * @size: Number of bytes to read
 * @offset: File offset
 *
 * Copies kernel buffer to user space using copy_to_user().
 */
static ssize_t example_dev_read(struct file *file,
				 char __user *user_buffer,
				 size_t size,
				 loff_t *offset)
{
	int bytes_to_copy = min(size, (size_t)BUFFER_SIZE);

	if (copy_to_user(user_buffer, device_buffer, bytes_to_copy) != 0) {
		printk(KERN_ERR "Failed to copy data to user space\n");
		return -EFAULT;
	}

	printk(KERN_INFO "Read %d bytes from device\n", bytes_to_copy);
	return bytes_to_copy;
}

/**
 * example_dev_write - Write to the device
 * @file: File structure pointer
 * @user_buffer: User-space buffer to read from
 * @size: Number of bytes to write
 * @offset: File offset
 *
 * Copies data from user space to kernel buffer.
 */
static ssize_t example_dev_write(struct file *file,
				  const char __user *user_buffer,
				  size_t size,
				  loff_t *offset)
{
	int bytes_to_copy = min(size, (size_t)(BUFFER_SIZE - 1));

	if (copy_from_user(device_buffer, user_buffer, bytes_to_copy) != 0) {
		printk(KERN_ERR "Failed to copy data from user space\n");
		return -EFAULT;
	}

	device_buffer[bytes_to_copy] = '\0';
	printk(KERN_INFO "Wrote %d bytes to device: %s\n", bytes_to_copy, device_buffer);
	return bytes_to_copy;
}

/**
 * example_dev_ioctl - Device control operations
 * @file: File structure pointer
 * @cmd: IOCTL command code
 * @arg: IOCTL argument (user-space pointer)
 *
 * Handles IOCTL commands for setting and getting device values.
 */
static long example_dev_ioctl(struct file *file,
			       unsigned int cmd,
			       unsigned long arg)
{
	unsigned int value;

	switch (cmd) {
	case IOCTL_SET_VALUE:
		if (copy_from_user(&value, (unsigned int __user *)arg, sizeof(value)) != 0) {
			printk(KERN_ERR "IOCTL SET: copy_from_user failed\n");
			return -EFAULT;
		}
		device_value = value;
		printk(KERN_INFO "Device value set to: %u\n", device_value);
		return 0;

	case IOCTL_GET_VALUE:
		if (copy_to_user((unsigned int __user *)arg, &device_value, sizeof(device_value)) != 0) {
			printk(KERN_ERR "IOCTL GET: copy_to_user failed\n");
			return -EFAULT;
		}
		printk(KERN_INFO "Device value retrieved: %u\n", device_value);
		return 0;

	default:
		printk(KERN_WARNING "Unknown IOCTL command: 0x%x\n", cmd);
		return -ENOTTY;
	}
}

/**
 * example_dev_open - Open the device
 * @inode: Inode structure pointer
 * @file: File structure pointer
 *
 * Called when the device is opened by a user-space process.
 */
static int example_dev_open(struct inode *inode, struct file *file)
{
	printk(KERN_INFO "Device opened\n");
	return 0;
}

/**
 * example_dev_release - Close the device
 * @inode: Inode structure pointer
 * @file: File structure pointer
 *
 * Called when the device is closed by a user-space process.
 */
static int example_dev_release(struct inode *inode, struct file *file)
{
	printk(KERN_INFO "Device closed\n");
	return 0;
}

static struct file_operations fops = {
	.open = example_dev_open,
	.release = example_dev_release,
	.read = example_dev_read,
	.write = example_dev_write,
	.unlocked_ioctl = example_dev_ioctl,
};

/**
 * example_dev_init - Module initialization
 *
 * Allocates a character device and registers it with the kernel.
 * Returns 0 on success, negative error code on failure.
 */
static int __init example_dev_init(void)
{
	printk(KERN_INFO "Initializing example device module\n");

	/* Allocate a device number */
	if (alloc_chrdev_region(&MKDEV(device_major, 0), 0, 1, DEVICE_NAME) < 0) {
		printk(KERN_ERR "Failed to allocate device number\n");
		return -1;
	}

	device_major = MAJOR(MKDEV(device_major, 0));
	printk(KERN_INFO "Device major number: %d\n", device_major);

	/* Create device class */
	device_class = class_create(DEVICE_NAME);
	if (IS_ERR(device_class)) {
		printk(KERN_ERR "Failed to create device class\n");
		unregister_chrdev_region(MKDEV(device_major, 0), 1);
		return -1;
	}

	/* Create device */
	device = device_create(device_class, NULL, MKDEV(device_major, 0),
			      NULL, DEVICE_NAME);
	if (IS_ERR(device)) {
		printk(KERN_ERR "Failed to create device\n");
		class_destroy(device_class);
		unregister_chrdev_region(MKDEV(device_major, 0), 1);
		return -1;
	}

	/* Initialize and add character device */
	cdev_init(&device_cdev, &fops);
	if (cdev_add(&device_cdev, MKDEV(device_major, 0), 1) < 0) {
		printk(KERN_ERR "Failed to add character device\n");
		device_destroy(device_class, MKDEV(device_major, 0));
		class_destroy(device_class);
		unregister_chrdev_region(MKDEV(device_major, 0), 1);
		return -1;
	}

	printk(KERN_INFO "Device module initialized successfully\n");
	return 0;
}

/**
 * example_dev_exit - Module cleanup
 *
 * Unregisters the device and releases all allocated resources.
 */
static void __exit example_dev_exit(void)
{
	printk(KERN_INFO "Cleaning up example device module\n");

	cdev_del(&device_cdev);
	device_destroy(device_class, MKDEV(device_major, 0));
	class_destroy(device_class);
	unregister_chrdev_region(MKDEV(device_major, 0), 1);

	printk(KERN_INFO "Device module unloaded\n");
}

module_init(example_dev_init);
module_exit(example_dev_exit);
